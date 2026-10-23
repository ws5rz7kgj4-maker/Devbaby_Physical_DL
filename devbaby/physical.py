"""Executed physical tasks. Simulator labels are confined to goal setup/scoring.

The policy receives RGB, sound, proprioception, its previous action, and an
explicit spatial/contact goal. It never receives live object poses, masses,
friction, delay, collision geometry, segmentation, or success labels.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import torch
from .world import World,ACTION_DIM,TRAIN_COMBOS
from .data import RandomExplorer
from .utils import derived_seed,atomic_json

DOMAINS=("iid","new_dynamics","new_delay","new_layout","joint_physics")
TASKS=("reach","touch","grasp_lift")

@dataclass(frozen=True)
class Goal:
    task: str
    position: tuple
    contact_position: tuple

class LearnedController:
    """CEM receding-horizon control using only the learned sensor dynamics."""
    def __init__(self,model,cfg,device,seed):
        self.model,self.cfg,self.device=model,cfg,device
        self.rng=np.random.default_rng(seed)
        self.state=None
        self.previous=np.zeros(ACTION_DIM,np.float32)
        self.mean=np.zeros((cfg.planner_horizon,ACTION_DIM),np.float32)

    @torch.no_grad()
    def __call__(self,obs,goal):
        m,cfg=self.model,self.cfg
        h,self.state=m.observe(obs,self.previous,self.device,self.state)
        acoustic=torch.as_tensor(obs["audio"],device=self.device).float()[None]
        if m.cfg.ablation=="no_audio":acoustic=torch.zeros_like(acoustic)
        hearing=m.hearing(acoustic)
        position=torch.tensor(goal.position,device=self.device)
        contact=torch.tensor(goal.contact_position,device=self.device)
        mean=self.mean.copy();std=np.ones_like(mean)*.75
        count=cfg.planner_candidates
        for _ in range(cfg.planner_iterations):
            candidates=np.clip(mean[None]+self.rng.normal(size=(count,*mean.shape))*std[None],-1,1).astype(np.float32)
            candidates[0]=mean
            candidates[:,:,4]=0  # head motion is irrelevant to these fixed-goal tasks
            if goal.task=="reach":candidates[:,:,3]=-1
            action=torch.as_tensor(candidates,device=self.device)
            pr=torch.as_tensor(obs["proprio"],device=self.device).float()[None].repeat(count,1)
            latent=h.repeat(count,1)
            cost=torch.zeros(count,device=self.device)
            for t in range(cfg.planner_horizon):
                a=m._actions(action[:,t])
                ha=torch.cat([latent,a],-1)
                visual=m.dynamics(ha)
                pr=pr+m.body(ha)
                held=pr[:,9].clamp(0,1)
                distance=(pr[:,:3]-position).square().sum(-1)
                if goal.task=="grasp_lift":
                    approach=(pr[:,:3]-contact).square().sum(-1)
                    # Explicit task objective: establish a grasp, then lift.
                    # This cost specifies the task; the movement predictor is learned.
                    stepcost=(1-held)*approach+held*distance+.08*(1-held)
                else:stepcost=distance
                cost+=stepcost*(3 if t==cfg.planner_horizon-1 else 1)
                x=torch.cat([visual,hearing.repeat(count,1),pr,a],-1).unsqueeze(1)
                y,_=m._memory(x,latent.unsqueeze(0))
                latent=y[:,0]
            cost+=.002*action[:,:,:3].square().mean((1,2))
            elite=torch.topk(cost,max(2,count//8),largest=False).indices.cpu().numpy()
            mean=candidates[elite].mean(0)
            std=candidates[elite].std(0).clip(.10,.8)
        chosen=mean[0].copy()
        self.mean=np.concatenate([mean[1:],mean[-1:]],axis=0)
        self.previous=chosen
        return chosen

def make_goal(world,task,seed):
    rng=np.random.default_rng(derived_seed(seed,"goal"))
    pos=np.asarray(world.call("getBasePositionAndOrientation",world.object_ids[0])[0])
    if task=="reach":
        target=rng.uniform([.15,-.32,.15],[.8,.32,.45])
        if world.condition in {"new_layout","joint_physics"} and abs(target[0]-.48)<.10 and abs(target[1]+.08)<.21:
            target[2]=max(target[2],.26)
    elif task=="touch":target=pos+np.array([0,0,.025])
    else:target=pos+np.array([0,0,.32])
    return Goal(task,tuple(target.tolist()),tuple((pos+np.array([0,0,.025])).tolist())),float(pos[2])

def reference_action(world,obs,goal,phase):
    """Privileged scripted feasibility reference; never called by learned policy."""
    a=np.zeros(ACTION_DIM,np.float32)
    pr=obs["proprio"];target=np.asarray(goal.position)
    if goal.task in {"touch","grasp_lift"} and not pr[9]:
        target=np.asarray(world.call("getBasePositionAndOrientation",world.object_ids[0])[0])+[0,0,.025]
        a[3]=1 if goal.task=="grasp_lift" else -1
    elif goal.task=="grasp_lift":a[3]=1
    else:a[3]=-1
    # The oracle may route above an obstacle. This is an upper reference,
    # not evidence that the learned agent planned such a route.
    if world.condition in {"new_layout","joint_physics"} and abs(pr[0]-target[0])>.10:
        if phase[0]==0:
            if pr[2]>.35:phase[0]=1
            else:target=np.array([pr[0],pr[1],.40])
        if phase[0]==1:
            if np.linalg.norm(pr[:2]-target[:2])<.07:phase[0]=2
            else:target=target.copy();target[2]=.40
    # Privileged reference anticipates queued actuator commands and drives the
    # internal servo target. This is deliberately unavailable to the learned actor.
    queued_target=world.target.copy()
    for pending in world.action_queue:
        queued_target=np.clip(queued_target+pending[:3]*world.motor_step,[-.1,-.62,.045],[1.1,.62,.65])
    if world.held_id is not None:
        target=target.copy()
        load=next(o['mass'] for o in world.objects if o['id']==world.held_id)
        target[2]+=load*abs(world.gravity)/world.motor_gain
    a[:3]=np.clip((target-queued_target)/world.motor_step,-1,1)
    return a

def execute_trial(model,cfg,speech,device,domain,task,trial,namespace,policy="learned"):
    seed=derived_seed(namespace,"physical",domain,task,trial)
    combo=TRAIN_COMBOS[trial%len(TRAIN_COMBOS)]
    with World(cfg,speech,seed,domain,objects=[combo],caregiver=False) as w:
        obs=w.observe();goal,initial_height=make_goal(w,task,seed)
        controller=LearnedController(model,cfg,device,derived_seed(seed,"planner")) if policy=="learned" else None
        random_policy=RandomExplorer(derived_seed(seed,"random"))
        phase=[0];history=[];actions=[];objects=[];success_frames=[];contacts=[]
        for _ in range(cfg.physics_steps):
            if policy=="learned":action=controller(obs,goal)
            elif policy=="random":action=random_policy(obs)
            elif policy=="oracle_reference":action=reference_action(w,obs,goal,phase)
            else:raise ValueError(policy)
            obs,_,ann=w.step(action)
            pos=np.asarray(w.call("getBasePositionAndOrientation",w.object_ids[0])[0])
            touching=any(x[2]==w.object_ids[0] for x in w.call("getContactPoints",bodyA=w.hand))
            distance=float(np.linalg.norm(obs["proprio"][:3]-np.asarray(goal.position)))
            if task=="reach":ok=distance<.08
            elif task=="touch":ok=touching
            else:ok=w.held_id==w.object_ids[0] and pos[2]>=initial_height+.14 and distance<.13
            success_frames.append(bool(ok));contacts.append(bool(touching))
            history.append(obs["proprio"].tolist());actions.append(action.tolist());objects.append(pos.tolist())
        # Fixed horizon, sustained success at the end. No cherry-picked best frame.
        success=bool(any(all(success_frames[i:i+3]) for i in range(len(success_frames)-2))) if task=="touch" else bool(all(success_frames[-3:]))
        return dict(domain=domain,task=task,trial=trial,policy=policy,seed=seed,success=success,
                    goal=goal.__dict__,initial_object_height=initial_height,final_distance=distance,
                    ever_succeeded=any(success_frames),success_frames=success_frames,
                    observed_proprio=history,issued_actions=actions,object_positions_for_scoring_only=objects,
                    contact_frames=contacts,scene=w.metadata())

def summarize_trials(rows):
    cells={}
    for domain in sorted({r["domain"] for r in rows}):
        cells[domain]={}
        for task in TASKS:
            r=[x for x in rows if x["domain"]==domain and x["task"]==task]
            if r:cells[domain][task]={"success_rate":float(np.mean([x["success"] for x in r])),"n":len(r),
                                    "mean_final_distance_m":float(np.mean([x["final_distance"] for x in r]))}
    ood=[v["success_rate"] for d,t in cells.items() if d!="iid" for v in t.values()]
    return dict(cells=cells,ood_macro_success=float(np.mean(ood)) if ood else None,
                n_trials=len(rows),goal_interface="explicit spatial/contact goals, not language instructions")

@torch.no_grad()
def physical_evaluate(model,cfg,speech,device,output,step,interactions,domains=DOMAINS,policy="learned",namespace=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    namespace=namespace or cfg.eval_namespace
    was_training=model.training;model.eval();rows=[]
    for domain in domains:
        for task in TASKS:
            for i in range(cfg.physics_trials):
                rows.append(execute_trial(model,cfg,speech,device,domain,task,i,namespace,policy))
    result=dict(update=step,interactions=interactions,physical=summarize_trials(rows),namespace=namespace,
                evaluation_mode="physics",weights_frozen=True,policy=policy)
    atomic_json(output/f"eval_{step:08d}.json",result)
    with open(output/f"trials_{step:08d}.jsonl","w") as f:
        for row in rows:f.write(json.dumps(row)+"\n")
    if was_training:model.train()
    return result
