from pathlib import Path
import typer
from rich import print
from .utils import load_yaml,write_jsonl,read_jsonl
from .data.discovery import discover_dataset_paths,maybe_extract_archive
from .data.adapters import build_manifest
from .data.audit import audit_samples
from .experiments import run_experiment

app=typer.Typer(add_completion=False)

@app.command()
def discover(dataset_root:str):
    root=Path(dataset_root);print(f'[bold]Dataset root:[/bold] {root}')
    for p in sorted(root.iterdir()):print(p.name)

@app.command()
def prepare(config:str='configs/project.yaml'):
    cfg=load_yaml(config);root=Path(cfg['dataset_root']);art=Path(cfg['artifact_root']);cache=art/'extracted'
    for ds,dcfg in cfg['datasets'].items():
        if not dcfg.get('enabled',True):continue
        found=discover_dataset_paths(root,dcfg['candidates'])
        if not found:print(f'[yellow]{ds}: not found[/yellow]');continue
        found=sorted(found,key=lambda p:(not p.is_dir(),len(str(p))));src=found[0]
        try:src=maybe_extract_archive(src,cache/ds)
        except Exception as e:print(f'[red]{ds}: extraction failed: {e}[/red]');continue
        samples=build_manifest(ds,src);mpath=art/'manifests'/f'{ds}.jsonl';write_jsonl([s.to_dict() for s in samples],mpath);rep=audit_samples(samples,art/'audits'/f'{ds}.json');print(f'[green]{ds}[/green]: {len(samples)} pairs, {rep["caption_count"]} captions -> {mpath}')

@app.command()
def experiment(exp_id:str,manifest:str,config:str='configs/project.yaml',outdir:str='results/manual_run',predictions:str='',limit:int=1000,expensive_limit:int=0):
    run_experiment(exp_id.upper(),load_yaml(config),manifest,outdir,predictions or None,limit,expensive_limit);print(f'Finished {exp_id}; artifacts: {outdir}')

@app.command('vlm-infer')
def vlm_infer(manifest:str,output:str,config:str='configs/project.yaml',split:str='test',limit:int=0):
    from .vlm.qwen import QwenPairCaptioner
    cfg=load_yaml(config);vc=cfg['vlm'];cap=QwenPairCaptioner(vc['model_name_or_path']);rows=[r for r in read_jsonl(manifest) if split=='all' or r['split']==split]
    if limit>0:rows=rows[:limit]
    out=[]
    for i,r in enumerate(rows,1):
        out.append({'image_id':r['image_id'],'prediction':cap.generate(r['t1_path'],r['t2_path'],vc['prompt'],vc['max_new_tokens'],vc['temperature'])})
        if i%20==0:write_jsonl(out,output);print(f'{i}/{len(rows)}')
    write_jsonl(out,output);print(f'Wrote {len(out)} predictions to {output}')

if __name__=='__main__':app()
