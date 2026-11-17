import argparse,random
from pathlib import Path
import torch
from ..utils import read_jsonl,load_yaml
from .common import load_qwen_with_lora,build_pair_inputs

def move(b,d):return {k:(v.to(d) if hasattr(v,'to') else v) for k,v in b.items()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/project.yaml'); ap.add_argument('--manifest',required=True); ap.add_argument('--output',required=True); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--aug_jsonl',default=None,help='optional extra training rows (same schema as the manifest, e.g. TemporalSwap-SFT rows from scripts/build_temporal_swap_data.py) appended to the train split; default None = unchanged baseline SFT')
    ap.add_argument('--log_every',type=int,default=0,help='print running loss every N samples (0 = silent, as before)'); a=ap.parse_args()
    cfg=load_yaml(a.config); tc=cfg['training']; prompt=cfg['vlm']['prompt']; random.seed(a.seed); torch.manual_seed(a.seed)
    rows=[r for r in read_jsonl(a.manifest) if r['split']=='train' and r['references']]
    if a.aug_jsonl: rows=rows+[r for r in read_jsonl(a.aug_jsonl) if r['references']]
    m,p=load_qwen_with_lora(tc['model_name_or_path'],tc['lora_r'],tc['lora_alpha'],tc['lora_dropout']); m.train(); opt=torch.optim.AdamW(m.parameters(),lr=float(tc['learning_rate'])); dev=next(m.parameters()).device; ga=int(tc['grad_accum']); step=0
    for _ in range(int(tc['epochs'])):
        random.shuffle(rows)
        for r in rows:
            b=move(build_pair_inputs(p,r['t1_path'],r['t2_path'],prompt,random.choice(r['references']),tc['max_length']),dev); loss=m(**b).loss/ga; loss.backward(); step+=1
            if a.log_every and step%a.log_every==0:print(f'step {step} loss {loss.item()*ga:.4f}',flush=True)
            if step%ga==0:opt.step();opt.zero_grad(set_to_none=True)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);m.save_pretrained(out);p.save_pretrained(out)
    if a.aug_jsonl or a.log_every:
        import json;(out/'run_config.json').write_text(json.dumps({'seed':a.seed,'manifest':a.manifest,'aug_jsonl':a.aug_jsonl,'n_train_rows':len(rows),'training_cfg':tc},indent=1,default=str))
if __name__=='__main__':main()
