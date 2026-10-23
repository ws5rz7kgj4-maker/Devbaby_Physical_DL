"""Frozen evaluation. All semantic labels live here, never in model training."""
from pathlib import Path
import json
import math
from dataclasses import replace
import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
from .world import World,CONDITIONS,COLORS,SHAPES,TRAIN_COMBOS,HELDOUT,ACTION_DIM
from .audio import query_features
from .data import collect_episode,episode_batch,RandomExplorer
from .utils import derived_seed,atomic_json

def wilson(k,n):
    if n==0: return [None,None]
    z=1.96
    c=(k/n+z*z/(2*n))/(1+z*z/n)
    w=z*math.sqrt((k/n)*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,c-w),min(1,c+w)]

def candidate_combos(target,rng,allowed=None):
    c,s=target
    cs=[x for x in COLORS if x!=c]
    ss=[x for x in SHAPES if x!=s]
    valid=[(cc,ss_) for cc in cs for ss_ in ss if allowed is None or all(x in allowed for x in [target,(c,ss_),(cc,s),(cc,ss_)])]
    otherc,others=valid[int(rng.integers(len(valid)))]
    choices=[target,(c,others),(otherc,s),(otherc,others)]
    order=rng.permutation(4)
    return [choices[i] for i in order],int(np.where(order==0)[0][0])

def prepare_evaluation(cfg,speech,root):
    # Every training variant is tested on the same evaluation distribution.
    cfg=replace(cfg,correspondence=1.0,caregiver_delay_steps=0,exploration="random",ablation="none")
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    if (root/"complete.json").exists(): return
    # Fixed evaluation seed independent of training seed: paired comparisons across models.
    for condition in CONDITIONS:
        dst=root/"grounding"/condition;dst.mkdir(parents=True,exist_ok=True)
        for i in range(cfg.eval_per_condition):
            path=dst/f"{i:05d}.npz"
            if path.exists(): continue
            rng=np.random.default_rng(derived_seed(cfg.eval_namespace,"eval",condition,i))
            options=sorted(HELDOUT) if condition in {"combination","joint"} else TRAIN_COMBOS
            target=options[i%len(options)]
            choices,correct=candidate_combos(target,rng,None if condition in {"combination","joint"} else set(TRAIN_COMBOS))
            imgs=[]
            scene_seed=derived_seed(cfg.eval_namespace,"eval-scene",condition,i)
            for combo in choices:
                with World(cfg,speech,scene_seed,condition,objects=[combo],caregiver=False) as w:
                    imgs.append(w.observe()["rgb"])
            text=["{c} {s}","look at the {c} {s}","there is a {c} {s}"][i%3].format(c=target[0],s=target[1])
            voice="en-gb-scotland" if condition in {"voice","joint"} else ["en-us","en"][i%2]
            wav=speech.synthesize(text,voice,[175,195,215][i%3],[35,50,65][i%3])
            np.savez_compressed(path,rgb=np.stack(imgs),audio=query_features(wav,cfg.control_hz,cfg.sample_rate),
                                correct=np.array(correct),target_color=np.array(list(COLORS).index(target[0])),
                                target_shape=np.array(SHAPES.index(target[1])))
            atomic_json(path.with_suffix(".json"),dict(target=target,choices=choices,correct=correct,text=text,voice=voice,scene_seed=scene_seed))
        for i in range(cfg.eval_dynamics_episodes):
            collect_episode(cfg,speech,root/"dynamics"/condition/f"{i:05d}",derived_seed(cfg.eval_namespace,"eval-dynamics",condition,i),condition)
    # Diagnostic learned representation probes. Labels used only by separate ridge readouts.
    images=[];colors=[];shapes=[]
    for i in range(cfg.probe_examples):
        combo=TRAIN_COMBOS[i%len(TRAIN_COMBOS)]
        with World(cfg,speech,derived_seed(cfg.eval_namespace,"probe-train",i),"iid",objects=[combo],caregiver=False) as w:
            images.append(w.observe()["rgb"])
        colors.append(list(COLORS).index(combo[0]));shapes.append(SHAPES.index(combo[1]))
    np.savez_compressed(root/"probe_train.npz",rgb=np.stack(images),color=np.array(colors),shape=np.array(shapes))
    # Same object/background across candidate event clips, with the event manipulated.
    eventroot=root/"events";eventroot.mkdir(exist_ok=True)
    for i in range(max(3,cfg.eval_per_condition//3)):
        clips=[];target=TRAIN_COMBOS[i%len(TRAIN_COMBOS)]
        for event in ["still","moving","falling"]:
            with World(cfg,speech,derived_seed(cfg.eval_namespace,"event-scene",i),"iid",objects=[target],caregiver=False) as w:
                oid=w.object_ids[0]
                if event=="falling":
                    pos,rot=w.call("getBasePositionAndOrientation",oid)
                    w.call("resetBasePositionAndOrientation",oid,[pos[0],pos[1],.55],rot)
                frames=[]
                for t in range(cfg.clip_steps):
                    if event=="moving":
                        pos,_=w.call("getBasePositionAndOrientation",oid)
                        force=w.objects[0]["mass"]*cfg.physics_hz*.5*(-1 if i%2 else 1)
                        w.call("applyExternalForce",oid,-1,[0,force,0],pos,p.WORLD_FRAME)
                    frames.append(w.observe()["rgb"])
                    w.step(np.zeros(ACTION_DIM,np.float32))
                clips.append(np.stack(frames))
        correct=i%3
        text=f"the {target[0]} {target[1]} is {['still','moving','falling'][correct]}"
        wav=speech.synthesize(text)
        np.savez_compressed(eventroot/f"{i:05d}.npz",rgb=np.stack(clips),audio=query_features(wav,cfg.control_hz,cfg.sample_rate),correct=np.array(correct))
    atomic_json(root/"complete.json",dict(conditions=list(CONDITIONS),n=cfg.eval_per_condition,namespace=cfg.eval_namespace,
                                        evaluation_labels_never_used_by_training=True))

def tensor_rgb(x,device):
    t=torch.as_tensor(x,device=device).float()/255
    return t.permute(0,3,1,2) if t.ndim==4 else t.permute(0,1,4,2,3)

@torch.no_grad()
def grounding(model,root,device):
    result={};trials=[]
    for condition in CONDITIONS:
        k=0;n=0
        for path in sorted((Path(root)/"grounding"/condition).glob("*.npz")):
            with np.load(path,allow_pickle=False) as f:
                v=model.grounding_images(tensor_rgb(f["rgb"],device))
                a=model.grounding_audio(torch.as_tensor(f["audio"],device=device).unsqueeze(0))
                scores=(v@a.T).flatten().cpu().numpy()
                guess=int(scores.argmax());correct=int(f["correct"])
            k+=guess==correct;n+=1
            trials.append(dict(condition=condition,trial=path.stem,correct=int(guess==correct),choice=guess,scores=scores.tolist()))
        result[condition]={"accuracy":k/n,"n":n,"correct":k,"chance":.25,"wilson95":wilson(k,n)}
    k=0;n=0
    for path in sorted((Path(root)/"events").glob("*.npz")):
        with np.load(path,allow_pickle=False) as f:
            v=model.grounding_images(tensor_rgb(f["rgb"],device))
            a=model.grounding_audio(torch.as_tensor(f["audio"],device=device).unsqueeze(0))
            guess=int((v@a.T).argmax());correct=int(f["correct"])
        k+=guess==correct;n+=1
    result["event_diagnostic"]={"accuracy":k/n,"n":n,"correct":k,"chance":1/3,"wilson95":wilson(k,n),
                                "scope":"controlled still/moving/falling clips; not held-out action composition"}
    return result,trials

@torch.no_grad()
def dynamics(model,root,device,cfg):
    result={}
    for condition in CONDITIONS:
        rows=[]
        for path in sorted((Path(root)/"dynamics"/condition).glob("*/observations.npz")):
            for start in range(0,cfg.episode_steps-cfg.clip_steps+1,cfg.clip_steps):
                b=episode_batch(path,device,start,cfg.clip_steps)
                out=model(b);t=cfg.clip_steps
                target=F.interpolate(b["rgb"][:,1:].reshape(t,3,64,64),size=32,mode="area")
                copy=F.interpolate(b["rgb"][:,:-1].reshape(t,3,64,64),size=32,mode="area")
                shuffled=dict(b);shuffled["actions"]=b["actions"].flip(1)
                alt=model(shuffled)
                body_target=b["proprio"][:,1:,:3]
                rows.append(dict(pixel_mse=float(F.mse_loss(out["pred_rgb"][0],target)),
                    pixel_copy_mse=float(F.mse_loss(copy,target)),
                    hand_position_mse=float(F.mse_loss(out["pred_body"][:,:,:3],body_target)),
                    hand_copy_mse=float(F.mse_loss(b["proprio"][:,:-1,:3],body_target)),
                    shuffled_action_hand_mse=float(F.mse_loss(alt["pred_body"][:,:,:3],body_target))))
        result[condition]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
        result[condition]["clips"]=len(rows)
    return result

@torch.no_grad()
def motor(model,cfg,speech,device):
    """Closed-loop reach using learned proprioceptive dynamics.

    Goal coordinates are supplied to the evaluation planner. This is NOT
    language-conditioned manipulation and never reads object coordinates.
    """
    rows=[]
    for trial in range(cfg.motor_trials):
        seed=derived_seed(cfg.eval_namespace,"motor",trial)
        rng=np.random.default_rng(seed)
        goal=rng.uniform([.15,-.32,.12],[.8,.32,.4]).astype(np.float32)
        for policy in ["learned","random","analytic_reference"]:
            with World(cfg,speech,seed,"iid",caregiver=False) as w:
                obs=w.observe();state=None;prev=np.zeros(ACTION_DIM,np.float32)
                explorer=RandomExplorer(seed)
                crng=np.random.default_rng(derived_seed(seed,"candidate"))
                initial=float(np.linalg.norm(obs["proprio"][:3]-goal))
                best=initial
                for _ in range(cfg.motor_steps):
                    if policy=="random": a=explorer(obs);a[3:]=0
                    elif policy=="analytic_reference":
                        a=np.zeros(ACTION_DIM,np.float32);a[:3]=np.clip((goal-obs["proprio"][:3])/.055,-1,1)
                    else:
                        h,state=model.observe(obs,prev,device,state)
                        ca=np.zeros((64,ACTION_DIM),np.float32);ca[:,:3]=crng.uniform(-1,1,(64,3));ca[0]=0
                        acts=torch.as_tensor(ca,device=device)
                        pred=torch.as_tensor(obs["proprio"],device=device)[None,:]+model.body(torch.cat([h.repeat(len(ca),1),model._actions(acts)],-1))
                        cost=(pred[:,:3]-torch.as_tensor(goal,device=device)).square().sum(-1)
                        a=ca[int(cost.argmin())]
                    obs,_,_=w.step(a);prev=a
                    best=min(best,float(np.linalg.norm(obs["proprio"][:3]-goal)))
                final=float(np.linalg.norm(obs["proprio"][:3]-goal))
                rows.append(dict(trial=trial,policy=policy,initial_distance_m=initial,final_distance_m=final,best_distance_m=best,success=final<.08))
    summary={}
    for policy in ["learned","random","analytic_reference"]:
        r=[x for x in rows if x["policy"]==policy]
        summary[policy]={"success_rate":float(np.mean([x["success"] for x in r])),
                         "final_distance_m":float(np.mean([x["final_distance_m"] for x in r])),"n":len(r)}
    summary["scope"]="3D hand reaching to explicit proprioceptive coordinate goals; no language task reward, no claim of instruction following"
    return summary,rows

@torch.no_grad()
def probes(model,root,device):
    def enc(images):
        parts=[]
        for i in range(0,len(images),32): parts.append(model.visual(tensor_rgb(images[i:i+32],device)).cpu().numpy())
        return np.concatenate(parts)
    with np.load(Path(root)/"probe_train.npz",allow_pickle=False) as f:
        x=enc(f["rgb"]);yc=f["color"];ys=f["shape"]
    # Standardization and ridge fitting use probe TRAIN images only.
    mean=x.mean(0);sd=x.std(0).clip(.05)
    x=np.c_[(x-mean)/sd,np.ones(len(x))]
    coefs=[]
    for y,n in [(yc,4),(ys,3)]:
        coefs.append(np.linalg.solve(x.T@x+10*np.eye(x.shape[1]),x.T@np.eye(n)[y]))
    result={}
    for condition in CONDITIONS:
        ims=[];labels=[]
        for path in sorted((Path(root)/"grounding"/condition).glob("*.npz")):
            with np.load(path,allow_pickle=False) as f:
                ims.append(f["rgb"][int(f["correct"])])
                labels.append([int(f["target_color"]),int(f["target_shape"])])
        z=enc(np.stack(ims));z=np.c_[(z-mean)/sd,np.ones(len(z))];labels=np.asarray(labels)
        result[condition]={name:float(np.mean((z@coefs[j]).argmax(1)==labels[:,j])) for j,name in enumerate(["color_accuracy","shape_accuracy"])}
    return {"note":"Supervised diagnostic ridge readouts on frozen features; NOT the infant's autonomous semantic performance.","scores":result}

def evaluate_all(model,cfg,speech,root,device,output,step,interactions):
    was_training=model.training;model.eval()
    with torch.no_grad():
        ground,trials=grounding(model,root,device)
        dyn=dynamics(model,root,device,cfg)
        mot,motor_rows=motor(model,cfg,speech,device)
        probe=probes(model,root,device)
    result=dict(update=step,interactions=interactions,grounding=ground,dynamics=dyn,motor=mot,probes=probe,
                consciousness_measurement=False,weights_frozen_during_evaluation=True)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/f"eval_{step:08d}.json",result)
    atomic_json(output/f"trials_{step:08d}.json",dict(grounding=trials,motor=motor_rows))
    if was_training: model.train()
    return result
