#!/usr/bin/env python3
"""Terminal launcher. No GPU/physics packages are installed on the submitting Mac.

Only the explicit `submit` command creates a paid HF Job. Repeating the same
command returns the recorded job. Tokens are sent as Jobs secrets, never saved.
"""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import sys
import time
import zipfile

ROOT=Path(__file__).resolve().parent
IMAGE="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
VARIANTS={
 "grounded":{},
 "q0":{"correspondence":0.0},
 "q25":{"correspondence":0.25},
 "q50":{"correspondence":0.5},
 "no_audio":{"ablation":"no_audio"},
 "no_action":{"ablation":"no_action"},
 "no_world_memory":{"ablation":"no_world_memory"},
 "shuffle_audio":{"ablation":"shuffle_audio"},
 "progress":{"exploration":"learning_progress"},
 "delayed":{"caregiver_delay_steps":12},
}

BOOTSTRAP=r'''
import os, pathlib, zipfile, subprocess, sys
from huggingface_hub import hf_hub_download
assert (3,10) <= sys.version_info[:2] <= (3,11), 'Use the supplied PyTorch 2.6 image (Python 3.10/3.11) for PyBullet wheels.'
archive=hf_hub_download(os.environ['DEVBABY_CODE_REPO'],os.environ['DEVBABY_CODE_PATH'],repo_type='dataset',revision=os.environ['DEVBABY_REVISION'])
root=pathlib.Path('/tmp/devbaby-source'); root.mkdir(exist_ok=True)
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert (root/name).resolve().is_relative_to(root.resolve()), 'Unsafe source zip'
    z.extractall(root)
subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check','-r',str(root/'requirements-runtime.txt')],check=True)
import torch
if os.environ['DEVBABY_REQUIRE_GPU']=='1' and not torch.cuda.is_available():
    raise RuntimeError('GPU job has no CUDA; refusing an accidental slow CPU run')
if os.environ['DEVBABY_CONFIG']=='formal.json':
    command=[sys.executable,'-u','-m','devbaby.suite','--protocol',str(root/'formal_protocol.json')]
else:
    command=[sys.executable,'-u','-m','devbaby.run','--config',str(root/'configs'/os.environ['DEVBABY_CONFIG'])]
command.extend(['--output','/tmp/devbaby-output','--device','cuda' if torch.cuda.is_available() else 'cpu',
         '--hf-repo',os.environ['DEVBABY_RESULT_REPO'],'--run-id',os.environ['DEVBABY_RUN_ID']])
for key,value in __import__('json').loads(os.environ['DEVBABY_OVERRIDES']).items():
    command.extend(['--'+key.replace('_','-'),str(value)])
if os.environ.get('DEVBABY_RESUME')=='1':command.append('--resume')
subprocess.run(command,cwd=root,check=True)
'''

def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with open(tmp,'w') as f:
        json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def source_bytes():
    buf=io.BytesIO()
    allowed={'.py','.json','.toml','.txt','.md'}
    top_level={'devbaby','configs','tests','hf_launch.py','hf_sweep.py','analyze_runs.py','START.py','formal_protocol.json','FORMAL_EXPERIMENT.md',
               'README.md','DATA_SCHEMA.md','SCIENTIFIC_SCOPE.md','VALIDATION.md','THIRD_PARTY.md',
               'requirements-runtime.txt','requirements-launcher.txt','pyproject.toml','Dockerfile','LICENSE'}
    with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(ROOT.rglob('*')):
            rel=f.relative_to(ROOT)
            if rel.parts[0] not in top_level:continue
            if not f.is_file() or any(x.startswith('.') or x in {'outputs','examples','verification','__pycache__'} for x in rel.parts):continue
            if f.suffix not in allowed and f.name not in {'Dockerfile','LICENSE'}:continue
            info=zipfile.ZipInfo(str(rel),date_time=(2026,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,f.read_bytes())
    data=buf.getvalue()
    return data,hashlib.sha256(data).hexdigest()

def duration(value):
    m=re.fullmatch(r'(\d+(?:\.\d+)?)(s|m|h|d)?',str(value))
    if not m:raise ValueError('Timeout must look like 30m, 3h or 10800')
    return float(m[1])*{'s':1,'m':60,'h':3600,'d':86400,None:1}[m[2]]

def api_client():
    from huggingface_hub import HfApi,get_token
    token=get_token()
    if not token:raise RuntimeError('First run: hf auth login (token needs repository write and Jobs permission).')
    return HfApi(token=token),token

def job_view(job):
    status=getattr(job.status,'stage',str(job.status))
    return dict(job_id=job.id,url=job.url,status=status)

def submit(args):
    data,digest=source_bytes()
    formal=args.profile=='formal'
    if formal and (args.seed!=0 or args.variant!='grounded'):
        raise ValueError('Formal seeds/variants are fixed in formal_protocol.json; do not override a single member at submission.')
    override={} if formal else {'seed':args.seed,**VARIANTS[args.variant]}
    cfg=json.loads((ROOT/'configs'/f'{args.profile}.json').read_text())
    cfg.update(override)
    timeout=args.timeout or {'smoke':'30m','pilot':'3h','research':'8h','formal':'24h'}[args.profile]
    if duration(timeout)<cfg.get('max_seconds',7200)+300:
        raise ValueError('Job timeout must exceed profile max_seconds by at least 300 seconds for setup and saving.')
    run_id=args.run_id or (f'formal-physical-{digest[:8]}' if formal else f'{args.profile}-{args.variant}-s{args.seed}-{digest[:8]}')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,95}',run_id):raise ValueError('Invalid run-id')
    spec=dict(profile=args.profile,variant=args.variant,seed=args.seed,run_id=run_id,source_sha256=digest,
              flavor=args.flavor,timeout=timeout,overrides=override,image=IMAGE,resume=args.resume)
    if args.dry_run:
        print(json.dumps(spec,indent=2));print('DRY RUN: no upload, no paid Job.');return spec
    api,token=api_client()
    owner=api.whoami()['name']
    code_repo=args.code_repo or f'{owner}/devbaby-source'
    result_repo=args.result_repo or f'{owner}/devbaby-results'
    spec.update(code_repo=code_repo,result_repo=result_repo)
    record_key=hashlib.sha256(f'{result_repo}/{run_id}'.encode()).hexdigest()[:20]
    records=ROOT/'.launch_records';records.mkdir(exist_ok=True)
    record_path=records/f'{record_key}.json'
    lock=records/f'{record_key}.lock'
    # Exclusive local lock avoids two terminals simultaneously submitting one run.
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError:raise RuntimeError(f'Another submission may be in progress. Inspect {record_path}; do not blindly retry a paid submission.')
    os.close(fd)
    try:
        old=json.loads(record_path.read_text()) if record_path.exists() else None
        if old:
            if old.get('source_sha256')!=digest or old.get('overrides')!=override:
                raise RuntimeError('This run-id belongs to different code/configuration. Choose a new --run-id.')
            if old.get('job_id'):
                job=api.inspect_job(job_id=old['job_id'],namespace=owner)
                view=job_view(job)
                if not args.resume or view['status'] not in {'COMPLETED','CANCELED','ERROR','DELETED'}:
                    atomic(records/'latest.json',{'record':str(record_path)})
                    print(json.dumps(view,indent=2));print('Existing Job reused; no new Job submitted.');return view
            elif old.get('phase')=='submitting':
                matches=[j for j in api.list_jobs(namespace=owner) if j.environment.get('DEVBABY_SUBMISSION_KEY')==old.get('submission_key')]
                if matches:
                    view=job_view(matches[0]);atomic(record_path,{**old,**view,'phase':'submitted'})
                    atomic(records/'latest.json',{'record':str(record_path)})
                    print(json.dumps(view,indent=2));return view
                raise RuntimeError('Previous submission has an uncertain outcome. Check HF Jobs and this record before starting a new run-id. No automatic retry was submitted.')
        checkpoint='suite_state.json' if formal else 'checkpoints/last.pt'
        if args.resume and not api.file_exists(repo_id=result_repo,repo_type='dataset',filename=f'runs/{run_id}/{checkpoint}'):
            raise RuntimeError('No uploaded checkpoint for --resume.')
        for repo in [code_repo,result_repo]:
            api.create_repo(repo_id=repo,repo_type='dataset',private=True,exist_ok=True)
            if not api.repo_info(repo_id=repo,repo_type='dataset').private:
                raise RuntimeError(f'Repository {repo} is public. Select a new private repo name explicitly.')
        code_path=f'source/{digest}.zip'
        commit=api.upload_file(path_or_fileobj=data,path_in_repo=code_path,repo_id=code_repo,repo_type='dataset',commit_message='DevBaby source package')
        submission_key=hashlib.sha256(f'{record_key}/{time.time_ns()}'.encode()).hexdigest()[:24]
        environment={'DEVBABY_CODE_REPO':code_repo,'DEVBABY_CODE_PATH':code_path,'DEVBABY_REVISION':commit.oid,
                     'DEVBABY_RESULT_REPO':result_repo,'DEVBABY_RUN_ID':run_id,'DEVBABY_CONFIG':f'{args.profile}.json',
                     'DEVBABY_OVERRIDES':json.dumps(override),'DEVBABY_RESUME':'1' if args.resume else '0',
                     'DEVBABY_REQUIRE_GPU':'0' if args.flavor.startswith('cpu') else '1',
                     'DEVBABY_SUBMISSION_KEY':submission_key,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4'}
        atomic(record_path,{**spec,'phase':'submitting','submission_key':submission_key,'revision':commit.oid})
        command=['bash','-lc','python -m pip install --disable-pip-version-check huggingface_hub==0.36.0 && python -c '+shlex.quote(BOOTSTRAP)]
        job=api.run_job(image=IMAGE,command=command,flavor=args.flavor,timeout=timeout,
                        env=environment,secrets={'HF_TOKEN':token},namespace=owner)
        view=job_view(job)
        atomic(record_path,{**spec,**view,'phase':'submitted','submission_key':submission_key,'revision':commit.oid})
        atomic(records/'latest.json',{'record':str(record_path)})
        print(json.dumps({**view,'run_id':run_id,'results':f'https://huggingface.co/datasets/{result_repo}/tree/main/runs/{run_id}'},indent=2))
        return view
    finally:
        lock.unlink(missing_ok=True)

def last_record(args):
    if args.record:return Path(args.record)
    p=ROOT/'.launch_records/latest.json'
    if not p.exists():raise FileNotFoundError('No local submission record. Supply --record PATH.')
    return Path(json.loads(p.read_text())['record'])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    s=sub.add_parser('submit')
    s.add_argument('--profile',choices=['smoke','pilot','research','formal'],default='formal')
    s.add_argument('--variant',choices=list(VARIANTS),default='grounded')
    s.add_argument('--seed',type=int,default=0)
    s.add_argument('--flavor',default='t4-small')
    s.add_argument('--timeout')
    s.add_argument('--run-id')
    s.add_argument('--code-repo')
    s.add_argument('--result-repo')
    s.add_argument('--dry-run',action='store_true')
    s.add_argument('--resume',action='store_true')
    for cmd in ['status','logs','download','cancel']:
        q=sub.add_parser(cmd);q.add_argument('--record')
        if cmd=='download':
            q.add_argument('--directory',default='outputs')
            q.add_argument('--all-data',action='store_true')
    args=parser.parse_args()
    if args.command=='submit':submit(args);return
    record=json.loads(last_record(args).read_text());api,_=api_client()
    if not record.get('job_id'):raise RuntimeError('Submission is unresolved; inspect HF Jobs before retrying.')
    if args.command=='status':print(json.dumps(job_view(api.inspect_job(job_id=record['job_id'])),indent=2))
    elif args.command=='logs':
        for line in api.fetch_job_logs(job_id=record['job_id']):print(line,end='' if line.endswith('\n') else '\n',flush=True)
    elif args.command=='cancel':api.cancel_job(job_id=record['job_id']);print('Cancellation requested.')
    else:
        from huggingface_hub import snapshot_download
        prefix=f"runs/{record['run_id']}"
        patterns=[f"{prefix}/**"]
        if record['profile']=='formal' and not args.all_data:
            patterns=[f"{prefix}/{x}" for x in ['protocol.json','suite_state.json','report.html','summary*','*.csv','*.png',
                      'members/*/config.json','members/*/status.json','members/*/metrics/*','members/*/final/**',
                      'members/*/learning_curves.*','members/*/report.html','members/*/preview.mp4',
                      'adaptation/*/state.json','adaptation/*/metrics/**','baselines/**']]
        snapshot_download(record['result_repo'],repo_type='dataset',allow_patterns=patterns,local_dir=args.directory)
        path=Path(args.directory)/'runs'/record['run_id']/'report.html'
        print('Report:',path.resolve())
        print('For the formal suite, add --all-data to also download every raw recording and model checkpoint.')

if __name__=='__main__':main()
