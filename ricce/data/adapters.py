from pathlib import Path
from collections import defaultdict
import json, csv, re
from .discovery import image_files, caption_files
from ..schema import Sample

def _split_guess(path):
    s=str(path).lower().replace('\\','/')
    for split in ['train','val','valid','validation','test']:
        if f'/{split}/' in s or f'_{split}_' in s or s.endswith(f'/{split}'):
            return 'val' if split in {'valid','validation'} else split
    return 'unknown'

def _time_role(path):
    parts=[x.lower() for x in path.parts]
    t1={'a','t1','time1','before','pre','old','im1','image1'}; t2={'b','t2','time2','after','post','new','im2','image2'}
    for part in reversed(parts[-5:]):
        tok=re.sub(r'[^a-z0-9]+','',part)
        if tok in t1 or tok.endswith('t1'): return 't1'
        if tok in t2 or tok.endswith('t2'): return 't2'
    return None

def _pair_key(path):
    stem=path.stem.lower()
    stem=re.sub(r'(^|[_-])(t1|t2|before|after|pre|post|old|new|a|b)([_-]|$)','_',stem)
    return re.sub(r'[^a-z0-9]+','',stem)

def _extract_texts(x):
    out=[]
    if isinstance(x,str): return [x]
    if isinstance(x,list):
        for v in x: out.extend(_extract_texts(v))
    elif isinstance(x,dict):
        for k in ['raw','caption','sentence','text','description']:
            if k in x: out.extend(_extract_texts(x[k]))
    return out

def _add_caption(by_filename,by_stem,key,text):
    if not key or not isinstance(text,str) or not text.strip(): return
    text=text.strip(); bn=Path(str(key)).name; stem=Path(bn).stem.lower()
    if text not in by_filename[bn]: by_filename[bn].append(text)
    if text not in by_stem[stem]: by_stem[stem].append(text)

def _consume(data,by_filename,by_stem):
    if isinstance(data,dict):
        if isinstance(data.get('images'),list): data=data['images']
        elif isinstance(data.get('annotations'),list): data=data['annotations']
        else:
            for k,v in data.items():
                for t in _extract_texts(v): _add_caption(by_filename,by_stem,k,t)
            return
    if isinstance(data,list):
        for row in data:
            if not isinstance(row,dict): continue
            key=next((str(row[k]) for k in ['filename','file_name','image','image_id','img','name','id'] if row.get(k) is not None),None)
            texts=[]
            for k in ['sentences','captions','caption','sentence','description','text']:
                if k in row: texts.extend(_extract_texts(row[k]))
            if key:
                for t in texts: _add_caption(by_filename,by_stem,key,t)

def _index_captions(root):
    bf=defaultdict(list); bs=defaultdict(list)
    for cf in caption_files(root):
        try:
            if cf.suffix.lower()=='.json': data=json.loads(cf.read_text(encoding='utf-8',errors='ignore'))
            elif cf.suffix.lower()=='.jsonl': data=[json.loads(x) for x in cf.read_text(encoding='utf-8',errors='ignore').splitlines() if x.strip()]
            elif cf.suffix.lower()=='.csv':
                with open(cf,newline='',encoding='utf-8',errors='ignore') as f: data=list(csv.DictReader(f))
            else: continue
            _consume(data,bf,bs)
        except Exception: continue
    return bf,bs

def _find_named_image(root,filename,split,role):
    cand=[]
    for p in Path(root).rglob(Path(filename).name):
        s=str(p).lower().replace('\\','/'); score=0
        if split and split.lower() in s: score+=2
        if f'/{role.lower()}/' in s: score+=3
        cand.append((score,p))
    return sorted(cand,key=lambda x:(-x[0],str(x[1])))[0][1] if cand else None

def levir_cc_manifest(root):
    root=Path(root).resolve()
    for jf in [p for p in root.rglob('*.json') if 'caption' in p.name.lower() or 'levir' in p.name.lower()]:
        try:
            data=json.loads(jf.read_text(encoding='utf-8',errors='ignore'))
            if not (isinstance(data,dict) and isinstance(data.get('images'),list)): continue
            out=[]
            for row in data['images']:
                fn=row.get('filename') or row.get('file_name'); split=row.get('split','unknown')
                if not fn: continue
                refs=_extract_texts(row.get('sentences',row.get('captions',[])))
                t1=_find_named_image(root,fn,split,'A') or _find_named_image(root,fn,split,'t1')
                t2=_find_named_image(root,fn,split,'B') or _find_named_image(root,fn,split,'t2')
                if t1 and t2:
                    out.append(Sample('LEVIR_CC',Path(fn).stem,'val' if split in {'valid','validation'} else split,str(t1),str(t2),refs,metadata={'caption_json':str(jf)}))
            if out: return out
        except Exception: pass
    return generic_pair_manifest('LEVIR_CC',root)

def generic_pair_manifest(dataset_name,root):
    root=Path(root).resolve(); imgs=image_files(root); bf,bs=_index_captions(root)
    grouped=defaultdict(lambda:{'t1':[],'t2':[],'unknown':[]})
    for p in imgs: grouped[_pair_key(p)][_time_role(p) or 'unknown'].append(p)
    samples=[]
    for key,g in grouped.items():
        t1=t2=None
        if g['t1'] and g['t2']: t1,t2=sorted(g['t1'])[0],sorted(g['t2'])[0]
        elif len(g['unknown'])==2: t1,t2=sorted(g['unknown'])
        else: continue
        split=_split_guess(t1); split=_split_guess(t2) if split=='unknown' else split
        refs=[]
        for k in [t1.name,t2.name]: refs+=bf.get(k,[])
        refs+=bs.get(t1.stem.lower(),[])+bs.get(t2.stem.lower(),[])
        seen=set(); refs=[x for x in refs if not (x in seen or seen.add(x))]
        samples.append(Sample(dataset_name,key or t1.stem,split,str(t1),str(t2),refs,metadata={'adapter':'generic_pair_manifest','root':str(root)}))
    return samples

def dubai_cc_manifest(root):
    root=Path(root).resolve()
    jsons=[p for p in root.rglob('*.json') if p.name.lower() in {'train_dubai_cc.json','test_dubai_cc.json','validation_dubai_cc.json'}]
    if not jsons: return []
    tile_dirs={}
    for p in sorted(root.rglob('*')):
        if p.is_dir() and p.name in {'500_2000','500_2010'}:
            # Prefer the RGB tile set (natural imagery) over PCA/other bands
            # for a captioning VLM; only fall back if RGB isn't available.
            if p.name not in tile_dirs or 'rgb' in str(p.parent).lower():
                tile_dirs[p.name]=p
    t1_dir=tile_dirs.get('500_2000'); t2_dir=tile_dirs.get('500_2010')
    if not (t1_dir and t2_dir): return []
    out=[]
    for jf in jsons:
        try:
            data=json.loads(jf.read_text(encoding='utf-8',errors='ignore'))
        except Exception:
            continue
        for row in data.get('images',[]):
            fn=row.get('filename')
            if not fn: continue
            t1=t1_dir/fn; t2=t2_dir/fn
            if not (t1.exists() and t2.exists()): continue
            refs=_extract_texts(row.get('sentences',[]))
            split=row.get('split','unknown'); split='val' if split in {'valid','validation'} else split
            out.append(Sample('DUBAI_CC',Path(fn).stem,split,str(t1),str(t2),refs,metadata={'caption_json':str(jf)}))
    return out

def build_manifest(dataset_name,root):
    if dataset_name=='LEVIR_CC': return levir_cc_manifest(root)
    if dataset_name=='DUBAI_CC':
        out=dubai_cc_manifest(root)
        return out if out else generic_pair_manifest(dataset_name,root)
    return generic_pair_manifest(dataset_name,root)
