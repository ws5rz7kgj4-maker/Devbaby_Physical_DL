#!/usr/bin/env python3
"""Explicit, sequential experiment matrix. Default is a plan, not a paid launch."""
import argparse
from argparse import Namespace
import json
import time
import hf_launch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['smoke','pilot','research'],default='pilot')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,4])
    p.add_argument('--variants',nargs='+',choices=list(hf_launch.VARIANTS),default=['grounded','q0','q25','q50','no_action','no_world_memory','shuffle_audio','progress'])
    p.add_argument('--flavor',default='t4-small')
    p.add_argument('--timeout')
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    plan=[{'profile':args.profile,'variant':v,'seed':s} for v in args.variants for s in args.seeds]
    print(json.dumps({'jobs':len(plan),'max_parallel':1,'plan':plan},indent=2))
    if not args.execute:
        print('No jobs submitted. Add --execute to submit this paid experiment matrix.');return
    api,_=hf_launch.api_client()
    for item in plan:
        a=Namespace(**item,flavor=args.flavor,timeout=args.timeout,run_id=None,code_repo=None,result_repo=None,dry_run=False,resume=False)
        view=hf_launch.submit(a)
        while True:
            job=api.inspect_job(job_id=view['job_id']);stage=job.status.stage
            print(f"{view['job_id']} {stage}",flush=True)
            if stage=='COMPLETED':
                from huggingface_hub import hf_hub_download
                record=json.loads(hf_launch.last_record(Namespace(record=None)).read_text())
                status_path=hf_hub_download(record['result_repo'],f"runs/{record['run_id']}/status.json",repo_type='dataset')
                state=json.loads(open(status_path).read())
                if state.get('status')!='completed':
                    raise RuntimeError(f"Job ended but experiment status is {state.get('status')}; sweep stopped.")
                break
            if stage in {'ERROR','CANCELED','DELETED'}:raise RuntimeError('Sweep stopped after a failed job. Inspect logs; no further jobs submitted.')
            time.sleep(30)

if __name__=='__main__':main()
