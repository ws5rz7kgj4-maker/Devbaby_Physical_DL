"""Measured physical-success curves, rates and adaptation sample efficiency."""
import csv
import html
import json
from pathlib import Path
import numpy as np

def normalized_auc(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<2 or x[-1]<=x[0]:return None
    return float(np.trapz(y,x)/(x[-1]-x[0]))

def curve_statistics(x,y,thresholds=(.5,.8)):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    slopes=[None]+[float(100*1000*(y[i]-y[i-1])/(x[i]-x[i-1])) for i in range(1,len(x))]
    crossings={}
    for threshold in thresholds:
        # Two consecutive scheduled measurements; no interpolation or fake zero.
        hit=next((int(x[i-1]) for i in range(1,len(y)) if y[i-1]>=threshold and y[i]>=threshold),None)
        crossings[str(threshold)]=hit
    return dict(normalized_auc=normalized_auc(x,y),growth_pp_per_1000_interactions=slopes,
                sustained_threshold_interactions=crossings,
                threshold_note="null means threshold was not observed at two consecutive scheduled evaluations")

def member_report(root,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root=Path(root);x=[r["interactions"] for r in rows]
    y=[r["physical"]["ood_macro_success"] for r in rows]
    stats=curve_statistics(x,y)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for task in ("reach","touch","grasp_lift"):
        values=[np.mean([c[task]["success_rate"] for d,c in r["physical"]["cells"].items() if d!="iid"]) for r in rows]
        axes[0].plot(x,values,".-",label=task)
    axes[0].plot(x,y,"k.-",label="OOD macro")
    axes[0].set(xlabel="Training environment interactions",ylabel="Executed success rate",ylim=(0,1))
    axes[0].legend();axes[0].grid(alpha=.2)
    axes[1].plot(x[1:],stats["growth_pp_per_1000_interactions"][1:],".-")
    axes[1].axhline(0,color="grey",linestyle="--")
    axes[1].set(xlabel="Training environment interactions",ylabel="Percentage points / 1,000 interactions")
    axes[1].grid(alpha=.2)
    fig.savefig(root/"learning_curves.png",dpi=150);plt.close(fig)
    with open(root/"learning_curves.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["interactions","update","domain","task","success_rate","n"]);writer.writeheader()
        for r in rows:
            for d,tasks in r["physical"]["cells"].items():
                for task,v in tasks.items():writer.writerow(dict(interactions=r["interactions"],update=r["update"],domain=d,task=task,success_rate=v["success_rate"],n=v["n"]))
    (root/"physical_curve_statistics.json").write_text(json.dumps(stats,indent=2))
    page=f'''<!doctype html><meta charset="utf-8"><title>Physical learning curves</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:20px}}img{{max-width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>Physical interaction success and growth</h1><p>These are executed simulator tasks, not speech-recognition scores.</p>
<img src="learning_curves.png"><p><a href="learning_curves.csv">Per-task CSV</a></p>
<p>Goals are explicit spatial/contact specifications. The body has a proximity-grasp primitive.
The movement predictor is learned from random experience; the task cost and CEM planner are supplied.</p>
<p>Development curves are descriptive; the separate final test uses fresh scene seeds and fixed training endpoints.</p>
<pre>{html.escape(json.dumps(stats,indent=2))}</pre>'''
    if (root/"preview.mp4").exists():page+='<video controls width="512" src="preview.mp4"></video>'
    (root/"report.html").write_text(page)
