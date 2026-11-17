from pathlib import Path
import json, hashlib, subprocess, platform, sys, yaml

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def dump_json(obj, path):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,ensure_ascii=False)

def read_jsonl(path):
    out=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): out.append(json.loads(line))
    return out

def write_jsonl(rows,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+'\n')

def sha256_file(path, chunk=1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            b=f.read(chunk)
            if not b: break
            h.update(b)
    return h.hexdigest()

def git_commit():
    try: return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception: return 'unknown'

def environment_info():
    d={'python':sys.version,'platform':platform.platform(),'git_commit':git_commit()}
    try:
        import torch
        d.update({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                  'cuda':torch.version.cuda,
                  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception: pass
    return d
