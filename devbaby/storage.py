from pathlib import Path
import json
import time
import zipfile
from .utils import sha256,atomic_json,safe_extract

def pack_paths(root,paths,destination):
    root,destination=Path(root),Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix(".tmp")
    with zipfile.ZipFile(tmp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for base in paths:
            base=Path(base)
            if base.is_dir():
                files=sorted(p for p in base.rglob("*") if p.is_file())
            else: files=[base]
            for f in files:z.write(f,f.relative_to(root))
    tmp.replace(destination)
    return {"file":str(destination.relative_to(root)),"sha256":sha256(destination),"bytes":destination.stat().st_size}

def sync(root,repo,run_id):
    if not repo:return
    from huggingface_hub import HfApi
    api=HfApi()
    patterns=["config.json","environment.json","status.json","data_manifest.json","report.html","learning_curves.*", "preview.mp4",
              "training_trace.jsonl","exposure.jsonl","metrics/*","checkpoints/*","data/shards/*","evaluation.zip","evaluation_reference.json","final/**","physical_curve_statistics.json","failure.txt"]
    for attempt in range(3):
        try:
            api.upload_folder(repo_id=repo,repo_type="dataset",folder_path=str(root),path_in_repo=f"runs/{run_id}",
                              allow_patterns=patterns,commit_message=f"DevBaby {run_id}: checkpoint and recorded experience")
            return
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)

def restore(root,repo,run_id):
    from huggingface_hub import snapshot_download
    root=Path(root)
    cache=root.parent/(root.name+"-restore")
    snapshot_download(repo,repo_type="dataset",allow_patterns=[f"runs/{run_id}/**"],local_dir=str(cache))
    import shutil
    src=cache/"runs"/run_id
    if not (src/"checkpoints"/"last.pt").exists():raise FileNotFoundError("No resumable checkpoint found in this run")
    shutil.copytree(src,root,dirs_exist_ok=True)
    for archive in [root/"evaluation.zip",*sorted((root/"data"/"shards").glob("*.zip"))]:
        if archive.exists():
            with zipfile.ZipFile(archive) as z:safe_extract(z,root)
