from pathlib import Path
import subprocess, shutil, zipfile, re

CAPTION_EXTS={'.json','.jsonl','.csv','.txt'}
IMAGE_EXTS={'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp'}

def discover_dataset_paths(dataset_root,candidates):
    root=Path(dataset_root).expanduser().resolve(); found=[]
    for name in candidates:
        p=root/name
        if p.exists(): found.append(p)
    norm=lambda s: re.sub(r'[^a-z0-9]+','',s.lower())
    cn={norm(x) for x in candidates}
    for p in root.iterdir():
        np=norm(p.name)
        if any(c in np or np in c for c in cn) and p not in found: found.append(p)
    return found

def maybe_extract_archive(path,cache_root):
    path=Path(path)
    if path.is_dir(): return path
    cache_root=Path(cache_root); cache_root.mkdir(parents=True,exist_ok=True)
    dest=cache_root/path.stem.replace('.tar','')
    if dest.exists() and any(dest.iterdir()): return dest
    dest.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()=='.zip':
        with zipfile.ZipFile(path) as z: z.extractall(dest)
        return dest
    if path.suffix.lower()=='.rar':
        exe=shutil.which('7z') or shutil.which('7zz') or shutil.which('unrar')
        if not exe: raise RuntimeError(f'RAR archive {path} requires 7z/7zz/unrar.')
        cmd=[exe,'x','-y',f'-o{dest}',str(path)] if 'unrar' not in Path(exe).name else [exe,'x','-o+',str(path),str(dest)]
        subprocess.run(cmd,check=True); return dest
    raise RuntimeError(f'Unsupported archive: {path}')

def image_files(root):
    return [p for p in Path(root).rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

def caption_files(root):
    out=[]
    for p in Path(root).rglob('*'):
        if p.is_file() and p.suffix.lower() in CAPTION_EXTS:
            n=p.name.lower()
            if any(k in n for k in ['caption','sentence','annotation','description','json']): out.append(p)
    return out
