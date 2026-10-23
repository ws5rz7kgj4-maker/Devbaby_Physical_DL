#!/usr/bin/env python3
"""Aggregate downloaded runs by independent training seed, using stdlib only."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

def bootstrap(values,seed=7,n=4000):
    if len(values)<2:return None,None
    rng=random.Random(seed)
    draws=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(n))
    return draws[int(.025*n)],draws[int(.975*n)]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory');p.add_argument('--output',default='aggregate.csv')
    p.add_argument('--threshold',type=float,default=.6);p.add_argument('--patience',type=int,default=3)
    args=p.parse_args()
    groups=defaultdict(dict);names={};onsets=[]
    for file in Path(args.directory).rglob('config.json'):
        root=file.parent
        if not (root/'metrics').is_dir():continue
        cfg=json.loads(file.read_text());seed=cfg['seed']
        exp={k:v for k,v in cfg.items() if k not in {'seed','threads','max_seconds','save_video'}}
        env=json.loads((root/'environment.json').read_text()) if (root/'environment.json').exists() else {}
        exp['source_archive']=env.get('source_archive','unknown')
        group=hashlib.sha256(json.dumps(exp,sort_keys=True).encode()).hexdigest()[:12]
        names[group]=exp
        rows=[json.loads(x.read_text()) for x in sorted((root/'metrics').glob('eval_*.json'))]
        for r in rows:
            for condition,v in r['grounding'].items():
                key=(group,r['interactions'],condition)
                if seed in groups[key]:raise ValueError(f'Duplicate seed {seed} for {key}; choose one run per seed.')
                groups[key][seed]=v['accuracy']
        for condition in ['iid','combination','layout','voice','physics','joint']:
            count=0;onset=None
            for j,r in enumerate(rows):
                count=count+1 if r['grounding'][condition]['accuracy']>=args.threshold else 0
                if count>=args.patience:
                    onset=rows[j-args.patience+1]['interactions'];break
            onsets.append(dict(group=group,seed=seed,condition=condition,threshold=args.threshold,patience=args.patience,onset_interactions=onset))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['group','interactions','condition','n_seeds','mean','seed_sd','bootstrap95_low','bootstrap95_high'])
        w.writeheader()
        for (group,interactions,condition),vals in sorted(groups.items()):
            v=list(vals.values());lo,hi=bootstrap(v)
            w.writerow(dict(group=group,interactions=interactions,condition=condition,n_seeds=len(v),mean=statistics.mean(v),
                            seed_sd=statistics.stdev(v) if len(v)>1 else None,bootstrap95_low=lo,bootstrap95_high=hi))
    out.with_suffix('.groups.json').write_text(json.dumps(names,indent=2))
    out.with_suffix('.onsets.json').write_text(json.dumps({'note':'Descriptive threshold crossings, not proof of emergence. Select thresholds before inspecting results; use continuous curves as primary outcomes.','rows':onsets},indent=2))
    print(f'Wrote {out}. Confidence intervals resample independent model seeds, not video frames.')
    if any(len(v)<5 for v in groups.values()):print('Some conditions have fewer than 5 independent seeds; interpret uncertainty cautiously.')

if __name__=='__main__':main()
