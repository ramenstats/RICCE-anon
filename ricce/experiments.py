from pathlib import Path
import pandas as pd
from .utils import read_jsonl,write_jsonl,dump_json,environment_info
from .ontology.tree import Taxonomy
from .parsing.hybrid import HybridParser
from .metrics.event import ricce_scores
from .metrics.conventional import evaluate_caption,cider_d_corpus
from .metrics.consensus import consensus_events
from .perturbations.generator import generate_all

def build_stack(cfg):
    taxes={k:Taxonomy.from_yaml(v) for k,v in cfg['ontology'].items()};p=cfg['parser'];parser=HybridParser(taxes['object'],taxes['action'],taxes['location'],taxes['quantity'],use_embeddings=p.get('use_embeddings',True),embedding_model=p.get('embedding_model'),similarity_threshold=float(p.get('similarity_threshold',0.58)));return taxes,parser

def e02(cfg,manifest,outdir):
    _,parser=build_stack(cfg);rows=read_jsonl(manifest);s={'captions':0,'object_mapped':0,'action_mapped':0,'location_mapped':0,'quantity_mapped':0}
    for r in rows:
        for cap in r['references']:
            s['captions']+=1;e=parser.parse(cap)[0]
            for k in ['object','action','location','quantity']:
                if getattr(e,k,None):s[f'{k}_mapped']+=1
    for k in ['object','action','location','quantity']:s[f'{k}_coverage']=s[f'{k}_mapped']/max(1,s['captions'])
    dump_json(s,Path(outdir)/'metrics.json');return s

def e04(cfg,manifest,outdir,limit=1000):
    taxes,parser=build_stack(cfg);weights=cfg['metric']['weights'];rows=read_jsonl(manifest);out=[];n=0
    for r in rows:
        for gt in r['references'][:1]:
            ge=parser.parse(gt)
            for p in generate_all(gt,taxes):
                ce=ricce_scores(parser.parse(p['text']),ge,taxes,weights);conv=evaluate_caption(p['text'],[gt],False);out.append({'image_id':r['image_id'],'type':p['type'],'gt':gt,'prediction':p['text'],**{k:v for k,v in ce.items() if k!='matches'},**conv})
            n+=1
            if n>=limit:break
        if n>=limit:break
    df=pd.DataFrame(out);df.to_csv(Path(outdir)/'per_sample_metrics.csv',index=False);df.groupby('type').mean(numeric_only=True).reset_index().to_csv(Path(outdir)/'summary.csv',index=False);dump_json({'rows':len(out),'types':sorted(set(x['type'] for x in out))},Path(outdir)/'metrics.json')

def e08(cfg,manifest,predictions,outdir,expensive_limit=0):
    # expensive_limit>0 computes BERTScore/Sentence-BERT (heavy CPU models)
    # for only the first N samples, since they are 10-100x slower than the
    # rest of the pipeline; 0 skips them entirely. CIDEr-D (cheap, corpus-
    # level) is always computed for every sample regardless.
    taxes,parser=build_stack(cfg);weights=cfg['metric']['weights'];m={r['image_id']:r for r in read_jsonl(manifest)};out=[]
    for i,pr in enumerate(read_jsonl(predictions)):
        if pr['image_id'] not in m:continue
        r=m[pr['image_id']];pe=parser.parse(pr['prediction']);ces=[ricce_scores(pe,parser.parse(g),taxes,weights) for g in r['references']]
        ce={k:(sum(x[k] for x in ces)/len(ces) if ces else float('nan')) for k in ['ce_precision','ce_recall','ce_f1','hallucination_rate','omission_rate']};conv=evaluate_caption(pr['prediction'],r['references'],expensive_limit>0 and i<expensive_limit);out.append({'image_id':r['image_id'],'split':r['split'],'prediction':pr['prediction'],**ce,**conv})
    ids=[x['image_id'] for x in out];preds=[x['prediction'] for x in out];refs_by_id={x['image_id']:m[x['image_id']]['references'] for x in out}
    cider_overall,cider_per_item=cider_d_corpus(ids,preds,refs_by_id)
    for x in out:x['cider_d']=cider_per_item.get(x['image_id'],float('nan'))
    df=pd.DataFrame(out);df.to_csv(Path(outdir)/'per_sample_metrics.csv',index=False);metrics={c:float(df[c].mean()) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])};metrics['cider_d_corpus']=cider_overall;dump_json(metrics,Path(outdir)/'metrics.json')

def e14(cfg,manifest,outdir,limit=1000):
    taxes,parser=build_stack(cfg);weights=cfg['metric']['weights'];out=[]
    for r in read_jsonl(manifest)[:limit]:out.append({'image_id':r['image_id'],'consensus_events':consensus_events([parser.parse(x) for x in r['references']],taxes,weights)})
    write_jsonl(out,Path(outdir)/'consensus_events.jsonl');dump_json({'samples':len(out)},Path(outdir)/'metrics.json')

def human_template(manifest,outdir):
    cols=['image_id','prediction','object_correctness','action_correctness','location_correctness','completeness','specificity','hallucination','overall_semantic_correctness','annotator_id'];pd.DataFrame(columns=cols).to_csv(Path(outdir)/'human_annotation_template.csv',index=False);dump_json({'status':'pending_human_annotations'},Path(outdir)/'metrics.json')

def run_experiment(exp_id,cfg,manifest,outdir,predictions=None,limit=1000,expensive_limit=0):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True);dump_json(environment_info(),outdir/'environment.json')
    if exp_id=='E02':return e02(cfg,manifest,outdir)
    if exp_id in {'E04','E05','E06'}:return e04(cfg,manifest,outdir,limit)
    if exp_id=='E07':return human_template(manifest,outdir)
    if exp_id in {'E08','E09','E18','E19'}:
        if not predictions:return dump_json({'status':'pending_predictions'},outdir/'metrics.json')
        return e08(cfg,manifest,predictions,outdir,expensive_limit)
    if exp_id=='E14':return e14(cfg,manifest,outdir,limit)
    dump_json({'status':'not_auto_executable','reason':'Requires external model outputs, human annotations, trained checkpoints, or a dedicated comparison configuration. No data is fabricated.'},outdir/'metrics.json')
