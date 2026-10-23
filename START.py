#!/usr/bin/env python3
"""Run from any folder: python3 /path/to/DevBaby_Physical_Final/START.py.

Default action submits ONE paid Hugging Face Job for the locked physical suite.
No local PyTorch/simulator install, no separate pilot, no automatic paid retries.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def run(command):
    subprocess.run([str(x) for x in command],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',choices=['run','plan','status','logs','download','resume','cancel'],default='run')
    parser.add_argument('--flavor',default='t4-small')
    parser.add_argument('--all-data',action='store_true')
    args=parser.parse_args()
    os.chdir(ROOT)
    if args.action=='plan':
        run([sys.executable,ROOT/'hf_launch.py','submit','--profile','formal','--flavor',args.flavor,'--dry-run'])
        return
    if sys.version_info<(3,10):raise SystemExit('需要 Python 3.10 或更新版本；Mac 上只安装云端启动器。')
    environment=ROOT/'.venv-final'
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():run([sys.executable,'-m','venv',environment])
    check=subprocess.run([str(python),'-c',"import importlib.metadata; assert importlib.metadata.version('huggingface_hub')=='0.36.0'"],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    if check.returncode:run([python,'-m','pip','install','--disable-pip-version-check','-r',ROOT/'requirements-launcher.txt'])
    authenticated=subprocess.run([str(python),'-c','from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 3)'],cwd=ROOT)
    if authenticated.returncode:
        executable=environment/('Scripts/hf.exe' if os.name=='nt' else 'bin/hf')
        run([executable,'auth','login'])
    command=[python,ROOT/'hf_launch.py']
    if args.action in {'run','resume'}:
        print('正式物理实验：16 个模型（2 条件 × 8 种子），合并为 1 个计费 Job。\nJob 上限 24 小时，完成即退出；超时后保存进度，不自动再开新 Job。\n主指标：新物理情境成功率、增长速度和适应学习曲线。',flush=True)
        command.extend(['submit','--profile','formal','--flavor',args.flavor])
        if args.action=='resume':command.append('--resume')
    else:
        command.append(args.action)
        if args.action=='download':
            command.extend(['--directory',str(ROOT/'outputs')])
            if args.all_data:command.append('--all-data')
    run(command)

if __name__=='__main__':main()
