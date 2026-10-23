"""Predeclared physical-success analysis. The independent unit is a model seed."""
import argparse
import csv
import html
import itertools
import json
from pathlib import Path
import numpy as np
from .physical_report import curve_statistics,normalized_auc
from .utils import atomic_json

def bootstrap_mean(values,seed=1931,draws=10000):
    v=np.asarray(values,float)
    if len(v)<2:return [None,None]
    r=np.random.default_rng(seed)
    means=r.choice(v,(draws,len(v)),replace=True).mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]

def sign_flip_p(differences):
    d=np.asarray(differences,float)
    if len(d)<2:return None
    if len(d)>16:raise ValueError("Exact enumeration limited to 16 seeds")
    actual=abs(d.mean());count=0
    for signs in itertools.product((-1,1),repeat=len(d)):
        count+=abs(np.mean(d*np.asarray(signs)))>=actual-1e-12
    return count/(2**len(d))

def holm(ps):
    result=[None]*len(ps);valid=sorted((p,i) for i,p in enumerate(ps) if p is not None)
    previous=0
    for rank,(p,i) in enumerate(valid):
        previous=max(previous,min(1.,p*(len(valid)-rank)));result[i]=previous
    return result

def load(path):return json.loads(Path(path).read_text())
def score(path):return load(path)["physical"]["ood_macro_success"]

def adaptation_gain_auc(budgets,values):
    values=np.asarray(values,float)
    return normalized_auc(budgets,values-values[0])

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(root):
    root=Path(root);locked=load(root/"protocol.json");p=locked["protocol"];cfg=locked["training_config"]
    if load(root/"suite_state.json").get("status")!="completed":raise ValueError("Cannot report a partial matrix as a completed formal experiment")
    seeds=p["seeds"];end=cfg["blocks"];step=end*cfg["updates_per_block"]
    member_curves={};curve_rows=[];growth_rows=[];thresholds=[]
    for variant in p["variants"]:
        trajectories=[]
        for seed in seeds:
            member=root/"members"/f"{variant}-s{seed}"
            metrics=[load(f) for f in sorted((member/"metrics").glob("eval_*.json"))]
            x=[m["interactions"] for m in metrics];y=[m["physical"]["ood_macro_success"] for m in metrics]
            if not x or x[-1]!=cfg["blocks"]*cfg["episodes_per_block"]*cfg["episode_steps"]:raise ValueError("Missing fixed endpoint")
            stats=curve_statistics(x,y);trajectories.append(y)
            thresholds.append(dict(variant=variant,seed=seed,**stats))
            for m in metrics:
                for domain,tasks in m["physical"]["cells"].items():
                    for task,v in tasks.items():curve_rows.append(dict(variant=variant,seed=seed,interactions=m["interactions"],domain=domain,task=task,success_rate=v["success_rate"],n=v["n"]))
            for i,slope in enumerate(stats["growth_pp_per_1000_interactions"]):
                if slope is not None:growth_rows.append(dict(variant=variant,seed=seed,interval_start=x[i-1],interval_end=x[i],pp_per_1000_interactions=slope))
        array=np.array(trajectories)
        member_curves[variant]={"x":x,"values":array,"mean":array.mean(0),"ci":np.array([bootstrap_mean(array[:,i]) for i in range(len(x))])}
    final_full=[];final_initial=[];final_ablation=[];final_rows=[]
    for seed in seeds:
        full=root/"members"/f"full-s{seed}"/"final"
        final_full.append(score(full/f"age_{end:04d}"/f"eval_{step:08d}.json"))
        final_initial.append(score(full/"age_0000"/"eval_00000000.json"))
        final_ablation.append(score(root/"members"/f"no_action-s{seed}"/"final"/f"age_{end:04d}"/f"eval_{step:08d}.json"))
        for condition,value in [("full",final_full[-1]),("initial",final_initial[-1]),("no_action",final_ablation[-1])]:
            final_rows.append(dict(seed=seed,condition=condition,ood_success=value))
    adaptation={};adapt_rows=[];adapt_aucs={};adapt_gain_aucs={};adapt_thresholds=[]
    for age in p["adaptation_checkpoint_blocks"]:
        sequences=[];aucs=[];gain_aucs=[]
        for seed in seeds:
            values=[]
            for budget in p["adaptation_budgets"]:
                by_domain=[]
                for domain in p["adaptation_domains"]:
                    directory=root/"adaptation"/f"s{seed}-age{age}-{domain}"/"metrics"/f"budget_{budget:06d}"
                    files=sorted(directory.glob("eval_*.json"))
                    if len(files)!=1:raise ValueError(f"Expected one adaptation checkpoint at {directory}")
                    value=score(files[0]);by_domain.append(value)
                    adapt_rows.append(dict(seed=seed,age_interactions=age*cfg["episodes_per_block"]*cfg["episode_steps"],domain=domain,extra_interactions=budget,success_rate=value))
                values.append(float(np.mean(by_domain)))
            sequences.append(values);aucs.append(normalized_auc(p["adaptation_budgets"],values))
            gain_aucs.append(adaptation_gain_auc(p["adaptation_budgets"],values))
            adapt_thresholds.append(dict(seed=seed,age_block=age,baseline_success=values[0],
                                         normalized_gain_auc=gain_aucs[-1],near_ceiling_at_start=values[0]>=.9,
                                         **curve_statistics(p["adaptation_budgets"],values)))
        a=np.asarray(sequences);adapt_aucs[age]=np.asarray(aucs);adapt_gain_aucs[age]=np.asarray(gain_aucs)
        adaptation[age]={"values":a,"mean":a.mean(0),"ci":np.array([bootstrap_mean(a[:,i]) for i in range(a.shape[1])])}
    differences=[np.array(final_full)-final_initial,np.array(final_full)-final_ablation,adapt_gain_aucs[end]-adapt_gain_aucs[0]]
    ps=[sign_flip_p(d) for d in differences];adjusted=holm(ps)
    tests=[]
    for name,d,pval,padj in zip(p["primary_family"],differences,ps,adjusted):
        tests.append(dict(contrast=name,n_seeds=len(d),mean_difference=float(d.mean()),paired_seed_ci95=bootstrap_mean(d),
                          exact_sign_flip_p=pval,holm_p=padj,positive_evidence=bool(not p["development_only"] and padj is not None and padj<.05 and d.mean()>0),per_seed_differences=d.tolist()))
    summary=dict(protocol_sha256=locked["sha256"],development_only=p["development_only"],primary_tests=tests,
                 final_full_mean=float(np.mean(final_full)),final_full_seed_ci95=bootstrap_mean(final_full),
                 final_initial_mean=float(np.mean(final_initial)),final_no_action_mean=float(np.mean(final_ablation)),
                 inference_unit="independent training seeds; paired CI and tests conditional on sampled test panel",
                 testing_assumption="Paired sign-flip tests assume exchangeability/symmetry of seed-level differences under the null.",
                 growth_note="Finite differences of fixed development checkpoints, in percentage points per 1,000 environment interactions. Negative growth is retained. Threshold non-attainment remains null/censored.",
                 adaptation_note="The third primary contrast uses baseline-subtracted adaptation AUC, separating additional learning gains from higher zero-shot starting success. Ceiling effects still limit cross-age interpretation; raw curves, slopes and starting levels are also reported.",
                 limitation="No power guarantee; spatial/contact goals and a task cost are supplied; task-wise failures and privileged-reference failures are reported.")
    atomic_json(root/"summary.json",summary);atomic_json(root/"summary_thresholds.json",dict(training=thresholds,adaptation=adapt_thresholds))
    write_csv(root/"success_curves.csv",curve_rows);write_csv(root/"growth_rates.csv",growth_rows)
    write_csv(root/"final_success.csv",final_rows);write_csv(root/"adaptation_curves.csv",adapt_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for variant,v in member_curves.items():
        axes[0,0].plot(v["x"],v["mean"],".-",label=variant)
        if len(seeds)>1:axes[0,0].fill_between(v["x"],v["ci"][:,0],v["ci"][:,1],alpha=.15)
        x=np.array(v["x"]);rates=100000*np.diff(v["values"],axis=1)/np.diff(x)
        axes[0,1].plot(x[1:],rates.mean(0),".-",label=variant)
        if len(seeds)>1:
            interval=np.array([bootstrap_mean(rates[:,i]) for i in range(rates.shape[1])])
            axes[0,1].fill_between(x[1:],interval[:,0],interval[:,1],alpha=.15)
    axes[0,0].set(title="Zero-shot transfer during development",xlabel="Base-training interactions",ylabel="OOD executed success",ylim=(0,1))
    axes[0,1].set(title="Physical success growth rate",xlabel="Base-training interactions",ylabel="Percentage points / 1,000 interactions")
    axes[0,1].axhline(0,color="gray",linestyle="--")
    for age,v in adaptation.items():
        label=f"base experience {age*cfg['episodes_per_block']*cfg['episode_steps']:,}"
        axes[1,0].plot(p["adaptation_budgets"],v["mean"],".-",label=label)
        if len(seeds)>1:axes[1,0].fill_between(p["adaptation_budgets"],v["ci"][:,0],v["ci"][:,1],alpha=.12)
    axes[1,0].set(title="Learning a new physical distribution",xlabel="Additional random interactions",ylabel="Held-out physical success",ylim=(0,1))
    ages=list(adapt_aucs);xx=[a*cfg["episodes_per_block"]*cfg["episode_steps"] for a in ages]
    means=[float(adapt_gain_aucs[a].mean()) for a in ages]
    axes[1,1].plot(xx,means,".-",label="adaptation gain AUC")
    if len(seeds)>1:
        ci=np.array([bootstrap_mean(adapt_gain_aucs[a]) for a in ages]);axes[1,1].fill_between(xx,ci[:,0],ci[:,1],alpha=.15)
    axes[1,1].axhline(0,color="gray",linestyle="--")
    axes[1,1].set(title="Additional learning gain at each developmental stage",xlabel="Base-training interactions",ylabel="Baseline-subtracted adaptation AUC")
    for ax in axes.flat:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(root/"physical_learning.png",dpi=160);plt.close(fig)
    baselines={policy:load(root/"baselines"/policy/"eval_00000000.json")["physical"] for policy in ["random","oracle_reference"]}
    atomic_json(root/"summary_baselines.json",baselines)
    page=f'''<!doctype html><meta charset="utf-8"><title>DevBaby physical experiments</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:36px auto;padding:20px;background:#f7f9fc;color:#172738}}img{{max-width:100%}}pre{{white-space:pre-wrap;background:white;padding:18px}}a{{color:#1b617d}}</style>
<h1>Physical interaction generalization and learning progress</h1>
<p>Completed fixed-budget experiment · {len(seeds)} training seeds · tasks: reach, touch, grasp-lift.</p>
<p>{'FUNCTIONAL TEST ONLY — NOT RESEARCH RESULTS.' if p['development_only'] else 'These are measured results; a flat or falling curve is a valid negative finding.'}</p>
<img src="physical_learning.png"><p><a href="success_curves.csv">Success by task/domain</a> · <a href="growth_rates.csv">Growth rates</a> · <a href="adaptation_curves.csv">Adaptation</a> · <a href="final_success.csv">Sealed final test</a></p>
<h2>Predeclared paired comparisons</h2><pre>{html.escape(json.dumps(summary,indent=2))}</pre>
<h2>Physical feasibility and random-control references</h2><pre>{html.escape(json.dumps(baselines,indent=2))}</pre>
<p>Goal coordinates/contact requirements and a CEM objective are supplied. The movement predictor is learned from raw sensory experience. The oracle reference uses simulator state solely to check task feasibility; its performance is not learned-agent performance.</p>
<p>All random training experiences, audible waveforms, annotations, checkpoints and adaptation experiences are retained in the result repository. No inference about consciousness or human developmental age is made.</p>'''
    (root/"report.html").write_text(page)
    return summary

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("directory");analyze(parser.parse_args().directory)
