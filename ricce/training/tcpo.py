import argparse,json,random,time
from pathlib import Path
import torch
from ..utils import read_jsonl,load_yaml
from ..ontology.tree import Taxonomy
from ..parsing.hybrid import HybridParser
from ..metrics.event import ricce_scores
from ..perturbations.generator import generate_all,generate_at_level
from contextlib import contextmanager
from .common import load_qwen_with_lora,load_qwen_with_lora_from,build_pair_inputs,build_pair_inputs_batch,sequence_logprob

@contextmanager
def _no_adapter(model):
    """Reference log-probs are computed by temporarily disabling the LoRA
    adapter on the *same* model (the standard TRL-DPOTrainer-style trick)
    instead of keeping a second full 7B model resident. With --init_from,
    the frozen base underneath the disabled adapter is already the merged
    SFT policy, so this is exactly "SFT as reference model" (design-review
    review Sec. 5.2 H2) at roughly half the memory footprint -- which is
    also what made the naive two-full-model version OOM on this shared GPU."""
    if hasattr(model,'disable_adapter'):
        with model.disable_adapter():
            yield
    else:
        yield

def move(b,d):return {k:(v.to(d) if hasattr(v,'to') else v) for k,v in b.items()}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/project.yaml')
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--mode',choices=['dpo','tcpo'],default='tcpo')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--init_from',default=None,help='SFT LoRA checkpoint to start the policy from and to use as the frozen reference model (internal design review, Sec. 5.2 H1/H2). If omitted, starts from and references the raw base model (the original, since-diagnosed-unstable setup).')
    ap.add_argument('--lr',type=float,default=None,help='overrides training.learning_rate; recommended range is 1e-6 to 5e-6 for preference optimization, well below the 2e-5 used for SFT.')
    ap.add_argument('--grad_accum',type=int,default=None,help='overrides training.grad_accum; target: an effective batch size of >=16.')
    ap.add_argument('--beta',type=float,default=None,help='overrides training.beta.')
    ap.add_argument('--warmup_steps',type=int,default=100)
    ap.add_argument('--max_grad_norm',type=float,default=1.0)
    ap.add_argument('--max_steps',type=int,default=0,help='0 = full epoch(s) as configured; >0 caps optimizer steps (for seed-stability sweeps).')
    ap.add_argument('--log_every',type=int,default=25)
    ap.add_argument('--ckpt_every',type=int,default=0,help='0 = only save at the end (original behavior). >0 saves a resumable checkpoint to <output>/step_<N>/ every N optimizer steps, so a run that later destabilizes does not lose everything -- added after a real run showed multi-signal instability (rising grad norm, degrading chosen log-prob, loss reversal) partway through with nothing saved to fall back on.')
    ap.add_argument('--diag_out',default=None,help='jsonl path for per-logged-step diagnostics; defaults to <output>/diagnostics.jsonl')
    ap.add_argument('--kl_coef',type=float,default=0.0,help='review "regularized DPO" fix: explicit sequence-level KL-to-reference penalty added to the loss, on top of the KL term already implicit in the DPO/TCPO log-ratio objective. Penalizes squared log-prob drift of the policy from the frozen reference on BOTH the chosen and rejected examples: kl_coef*0.5*mean((lp_chosen-ref_chosen)^2+(lp_rejected-ref_rejected)^2). 0.0 (default) reproduces the original, unregularized objective exactly.')
    ap.add_argument('--severity',choices=['taxonomy','random','uniform','clipped','normalized','inverse'],default='taxonomy',help="Only meaningful when --mode tcpo. 'taxonomy' (default): the original TCPO severity weight w=1+lambda*(1-CE-F(rejected,chosen)), i.e. genuinely harder-to-distinguish (taxonomically closer) negatives get up-weighted. 'random': the ablation that proves the *taxonomy signal* itself matters, not just having some non-uniform weight -- draws the severity input from an independent uniform random number instead of the real CE-F distance for that pair, so the weight is decoupled from actual difficulty. 'uniform': the TCPO-without-severity ablation -- weight is fixed at 1.0 for every pair. 'clipped' (NEW, v2 A2): same taxonomy severity input as 'taxonomy', but the resulting weight is clamped to [severity_clip_lo,severity_clip_hi] before use, to test whether the (unweighted-baseline-matching) collapse seen with plain 'taxonomy' weighting is being driven by extreme per-pair weights destabilizing training. 'normalized' (NEW, v2 A2): same taxonomy severity input, but weights are renormalized within each micro-batch to a fixed mean (severity_norm_target, default 1.0) before use, isolating relative (within-batch) severity ordering from absolute weight magnitude. 'inverse' (v2 A6 causal ablation negative control): w=1+lambda*(1-(1-CE-F)) = 1+lambda*CE-F, i.e. deliberately rewards taxonomically *easy* (already well-separated) pairs with the higher weight instead of hard ones -- a sanity-check wrong-direction control against which true 'taxonomy' severity must be compared under otherwise-identical conditions.")
    ap.add_argument('--severity_clip_lo',type=float,default=0.5,help="Lower clamp bound for --severity clipped.")
    ap.add_argument('--severity_clip_hi',type=float,default=2.0,help="Upper clamp bound for --severity clipped.")
    ap.add_argument('--severity_norm_target',type=float,default=1.0,help="Target per-micro-batch mean weight for --severity normalized.")
    ap.add_argument('--neg_strategy',choices=['random','hardest','curriculum'],default='random',help="How the rejected caption is chosen among the candidate taxonomy perturbations of the same reference. 'random' (default, original behavior): uniform choice among all non-paraphrase perturbation types. 'hardest': harder-negative mining -- scores every candidate's CE-F similarity to the chosen caption and keeps the one with the *highest* similarity (smallest taxonomy distance), i.e. the hardest one for the model to tell apart from the chosen caption, instead of a uniformly random pick that is often an easy structural corruption (e.g. hallucination/omission). 'curriculum' (NEW, v2 A4 hard-negative curriculum): samples the rejected caption's difficulty level from ricce.perturbations.generator.CURRICULUM_LEVEL (1=ancestor/coarse .. 6=mixed multi-axis) on a schedule that shifts from easy to hard over the course of training (see --curriculum_schedule), so the preference signal must increasingly come from genuine taxonomic/semantic correctness rather than superficial lexical/fluency cues. Level 4 (source-target reversal) is realized by swapping the (t1,t2) image order for the rejected item while keeping the caption text byte-identical to the chosen caption -- the hardest possible lexical-overlap control, since the two texts are literally the same string.")
    ap.add_argument('--curriculum_schedule',choices=['linear','uniform'],default='linear',help="Only with --neg_strategy curriculum. 'linear': difficulty level sampled increases linearly from 1 to 6 across training progress (opt_step/total_planned_steps). 'uniform': difficulty level sampled uniformly at random from 1-6 every pair, independent of training progress (a fixed-difficulty-mix ablation of the curriculum schedule itself).")
    ap.add_argument('--batch_size',type=int,default=1,help="Proper batching: number of (chosen,rejected) pairs collated into one real batched forward/backward call. The original loop always used micro-batch=1 and relied only on --grad_accum to reach an effective batch size, meaning single hard examples could dominate an individual gradient step; a true multi-sample batch (via build_pair_inputs_batch) averages the loss over B independently-sampled pairs before each accumulation step, which is the standard batching fix and also cuts wall-clock by amortizing Python/model-call overhead. 1 (default) reproduces the original per-sample loop exactly.")
    a=ap.parse_args()
    cfg=load_yaml(a.config);tc=cfg['training'];prompt=cfg['vlm']['prompt'];random.seed(a.seed);torch.manual_seed(a.seed)
    taxes={k:Taxonomy.from_yaml(v) for k,v in cfg['ontology'].items()};parser=HybridParser(taxes['object'],taxes['action'],taxes['location'],taxes['quantity'],use_embeddings=False);weights=cfg['metric']['weights']
    rows=[r for r in read_jsonl(a.manifest) if r['split']=='train' and r['references']]

    lr=a.lr if a.lr is not None else float(tc['learning_rate'])
    ga=a.grad_accum if a.grad_accum is not None else int(tc['grad_accum'])
    beta=a.beta if a.beta is not None else float(tc['beta'])
    lam=float(tc['tcpo_lambda'])

    if a.init_from:
        model,proc=load_qwen_with_lora_from(tc['model_name_or_path'],a.init_from,tc['lora_r'],tc['lora_alpha'],tc['lora_dropout'])
    else:
        model,proc=load_qwen_with_lora(tc['model_name_or_path'],tc['lora_r'],tc['lora_alpha'],tc['lora_dropout'])

    opt=torch.optim.AdamW(model.parameters(),lr=lr)
    dev=next(model.parameters()).device
    total_planned_steps=max(1,(len(rows)*int(tc['epochs']))//ga)
    def lr_scale(step):
        if step<a.warmup_steps:return (step+1)/max(1,a.warmup_steps)
        return 1.0
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda=lr_scale)

    def _curriculum_level(progress):
        """progress in [0,1] (opt_step/total_planned_steps). 'linear': ramps
        1->6 over training so early steps see coarse, easy-to-spot negatives
        and later steps see the hardest (mixed multi-axis / source-target)
        negatives -- standard curriculum-learning shaping applied to the A4
        hard-negative ladder. 'uniform': ignores progress, level ~ U{1..6}."""
        if a.curriculum_schedule=='uniform':return random.randint(1,6)
        lvl=1+int(progress*6)
        return max(1,min(6,lvl))

    def pick_rejected(chosen,progress=0.0):
        """Rejected-sample selection. Returns (text, ce_f1_vs_chosen, swap_images).
        --neg_strategy hardest (harder-negative mining) scores
        every candidate perturbation's CE-F similarity to the chosen caption
        and keeps the closest one -- the hardest for the model to
        discriminate -- instead of a uniform random pick, which often lands
        on an easy structural corruption (e.g. P8_hallucination) that the
        model can reject on cheap surface cues alone. --neg_strategy
        curriculum (v2 A4) instead samples a target difficulty
        LEVEL (1..6, ricce.perturbations.generator.CURRICULUM_LEVEL) per
        --curriculum_schedule, and draws a candidate at exactly that level;
        level 4 (source-target reversal) has no *text* candidate -- it is
        realized by keeping the caption text byte-identical to `chosen`
        (ce_f1=1.0, the hardest possible lexical-overlap case) and instead
        swapping which image is presented as T1 vs T2 for the rejected item,
        signalled via the returned swap_images flag. Always returns the real
        taxonomy-distance CE-F score for the pair actually used, independent
        of --severity, so it can be logged for diagnostics even when the
        weighting itself is randomized, clipped, normalized, inverted, or
        disabled."""
        if a.neg_strategy=='curriculum':
            lvl=_curriculum_level(progress)
            if lvl==4:
                return chosen,1.0,True
            cands=generate_at_level(chosen,taxes,lvl)
            if not cands:
                # this caption has no candidate at the sampled level (e.g. no
                # object mention for an ancestor-level corruption) -- fall
                # back to any non-identity, non-paraphrase perturbation
                # rather than dropping the pair.
                cands=[x for x in generate_all(chosen,taxes) if x['type'] not in ('P0_original','P0_paraphrase') and x['text']!=chosen]
            if not cands:return None,None,False
            x=random.choice(cands)
            return x['text'],ricce_scores(parser.parse(x['text']),parser.parse(chosen),taxes,weights)['ce_f1'],False
        negs=[x for x in generate_all(chosen,taxes) if x['type'] not in ('P0_original','P0_paraphrase') and x['text']!=chosen]
        if not negs:return None,None,False
        if a.neg_strategy=='hardest':
            scored=[(x['text'],ricce_scores(parser.parse(x['text']),parser.parse(chosen),taxes,weights)['ce_f1']) for x in negs]
            t,c=max(scored,key=lambda t:t[1]);return t,c,False
        x=random.choice(negs)
        return x['text'],ricce_scores(parser.parse(x['text']),parser.parse(chosen),taxes,weights)['ce_f1'],False

    def batched(seq,bs):
        buf=[]
        for x in seq:
            buf.append(x)
            if len(buf)>=bs:yield buf;buf=[]
        if buf:yield buf

    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    diag_path=Path(a.diag_out) if a.diag_out else out/'diagnostics.jsonl'
    diag_f=open(diag_path,'a')
    curriculum_denom=max(1,a.max_steps if a.max_steps else total_planned_steps)
    micro_step=0;opt_step=0;t0=time.time();accum_loss=0.0;accum_kl=0.0;stop=False
    for _ in range(int(tc['epochs'])):
        if stop:break
        random.shuffle(rows)
        for rb in batched(rows,a.batch_size):
            chosen_items=[];rejected_items=[];ce_f1s=[]
            progress=min(1.0,opt_step/curriculum_denom)
            for r in rb:
                chosen=random.choice(r['references'])
                rejected,ce_f1,swap_images=pick_rejected(chosen,progress)
                if rejected is None:continue
                chosen_items.append((r['t1_path'],r['t2_path'],chosen))
                if swap_images:
                    rejected_items.append((r['t2_path'],r['t1_path'],rejected))
                else:
                    rejected_items.append((r['t1_path'],r['t2_path'],rejected))
                ce_f1s.append(ce_f1)
            if not chosen_items:continue
            ch=move(build_pair_inputs_batch(proc,chosen_items,prompt,tc['max_length']),dev)
            rj=move(build_pair_inputs_batch(proc,rejected_items,prompt,tc['max_length']),dev)
            lpch=sequence_logprob(model,ch);lprj=sequence_logprob(model,rj)
            with torch.no_grad(),_no_adapter(model):rch=sequence_logprob(model,ch);rrj=sequence_logprob(model,rj)
            delta=(lpch-lprj)-(rch-rrj)
            if a.mode=='tcpo':
                if a.severity=='uniform':
                    w=torch.ones(len(ce_f1s),device=dev,dtype=delta.dtype)
                elif a.severity=='inverse':
                    # v2 A6 causal ablation negative control: same
                    # taxonomy CE-F input as 'taxonomy', but the weight
                    # formula is deliberately flipped to reward *easy*
                    # (already well-separated, high CE-F-distance) pairs
                    # instead of hard ones -- w=1+lambda*s instead of
                    # w=1+lambda*(1-s). If this performs about as well as
                    # true 'taxonomy' severity, that is evidence severity
                    # *direction* isn't doing anything either.
                    w=torch.tensor([1.0+lam*max(0.0,min(1.0,s)) for s in ce_f1s],device=dev,dtype=delta.dtype)
                else:
                    sev_input=ce_f1s if a.severity in ('taxonomy','clipped','normalized') else [random.random() for _ in ce_f1s]
                    w=torch.tensor([1.0+lam*max(0.0,min(1.0,1.0-s)) for s in sev_input],device=dev,dtype=delta.dtype)
                    if a.severity=='clipped':
                        w=w.clamp(a.severity_clip_lo,a.severity_clip_hi)
                    elif a.severity=='normalized':
                        w=w*(a.severity_norm_target/w.mean().clamp_min(1e-8))
            else:
                w=torch.ones(delta.shape[0],device=dev,dtype=delta.dtype)
            pref_loss=-w*torch.nn.functional.logsigmoid(beta*delta)
            kl_pen=0.5*((lpch-rch).pow(2)+(lprj-rrj).pow(2)).mean() if a.kl_coef>0 else torch.zeros((),device=dev)
            loss=pref_loss.mean()+a.kl_coef*kl_pen
            (loss/ga).backward()
            accum_loss+=float(loss.detach());accum_kl+=float(kl_pen.detach());micro_step+=1
            if micro_step%ga==0:
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),a.max_grad_norm)
                opt.step();sched.step();opt.zero_grad(set_to_none=True);opt_step+=1
                if opt_step%a.log_every==0:
                    ce_f1_mean=sum(ce_f1s)/len(ce_f1s) if ce_f1s else None
                    rec={'opt_step':opt_step,'micro_step':micro_step,'elapsed_s':round(time.time()-t0,1),
                         'loss':round(accum_loss/ga,6),'kl_penalty':round(accum_kl/ga,6),'lr':sched.get_last_lr()[0],
                         'grad_norm':round(float(grad_norm),4),
                         'chosen_logp':round(float(lpch.mean()),4),'rejected_logp':round(float(lprj.mean()),4),
                         'ref_chosen_logp':round(float(rch.mean()),4),'ref_rejected_logp':round(float(rrj.mean()),4),
                         'reward_margin':round(float(delta.mean()),4),
                         'kl_proxy_chosen':round(float((lpch-rch).mean()),4),'kl_proxy_rejected':round(float((lprj-rrj).mean()),4),
                         'ce_f1_rejected_vs_chosen':(round(ce_f1_mean,4) if ce_f1_mean is not None else None),
                         'weight':round(float(w.mean()),4),'batch_size':a.batch_size,'neg_strategy':a.neg_strategy,
                         'severity':(a.severity if a.mode=='tcpo' else None),'kl_coef':a.kl_coef}
                    diag_f.write(json.dumps(rec)+'\n');diag_f.flush()
                    print('DIAG',json.dumps(rec),flush=True)
                if a.ckpt_every and opt_step%a.ckpt_every==0:
                    ckpt_dir=out/f'step_{opt_step}'
                    model.save_pretrained(ckpt_dir);proc.save_pretrained(ckpt_dir)
                    print('CHECKPOINT',str(ckpt_dir),flush=True)
                accum_loss=0.0;accum_kl=0.0
                if a.max_steps and opt_step>=a.max_steps:stop=True;break
    diag_f.close()
    model.save_pretrained(out);proc.save_pretrained(out)
    dump={'mode':a.mode,'seed':a.seed,'init_from':a.init_from,'lr':lr,'grad_accum':ga,'beta':beta,
          'warmup_steps':a.warmup_steps,'max_grad_norm':a.max_grad_norm,'opt_steps':opt_step,'micro_steps':micro_step,
          'kl_coef':a.kl_coef,'severity':a.severity,'neg_strategy':a.neg_strategy,'batch_size':a.batch_size,
          'effective_batch_size':a.batch_size*ga,'max_steps':a.max_steps,
          'severity_clip_lo':a.severity_clip_lo,'severity_clip_hi':a.severity_clip_hi,
          'severity_norm_target':a.severity_norm_target,'curriculum_schedule':a.curriculum_schedule}
    (out/'run_config.json').write_text(json.dumps(dump,indent=2))
if __name__=='__main__':main()
