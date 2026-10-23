"""One cloud job: fixed physical-learning protocol, resume, final test, adaptation."""
import argparse
from dataclasses import asdict,replace
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import numpy as np
import torch
from .config import Config
from .audio import Speech
from .model import Baby,losses
from .data import collect_episode,Replay
from .run import train,save_checkpoint
from .physical import physical_evaluate
from .storage import pack_paths
from .utils import atomic_json,derived_seed,seed_all,device_for

def read_json(path):return json.loads(Path(path).read_text())

def push(root,repo,run_id,patterns):
    if not repo:return
    from huggingface_hub import HfApi
    for attempt in range(3):
        try:
            HfApi().upload_folder(repo_id=repo,repo_type="dataset",folder_path=str(root),path_in_repo=f"runs/{run_id}",
                                allow_patterns=patterns,commit_message=f"Physical experiment {run_id}")
            return
        except Exception:
            if attempt==2:raise
            time.sleep(2**attempt)

def restore_summary(root,repo,run_id):
    from huggingface_hub import snapshot_download
    cache=root.parent/(root.name+"-index")
    patterns=["protocol.json","suite_state.json","report.html","summary*","*.csv","*.png","failure.txt",
              "members/*/config.json","members/*/status.json","members/*/metrics/*","members/*/final/*",
              "adaptation/*/state.json","adaptation/*/metrics/**","baselines/**"]
    snapshot_download(repo,repo_type="dataset",allow_patterns=[f"runs/{run_id}/{x}" for x in patterns],local_dir=cache)
    src=cache/"runs"/run_id
    if not (src/"protocol.json").exists():raise FileNotFoundError("No saved formal protocol")
    shutil.copytree(src,root,dirs_exist_ok=True)

def ensure_file(root,relative,repo,run_id):
    path=root/relative
    if path.exists():return path
    if not repo:raise FileNotFoundError(path)
    from huggingface_hub import hf_hub_download
    source=hf_hub_download(repo,f"runs/{run_id}/{relative}",repo_type="dataset")
    path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,path)
    return path

def load_model(path,device):
    ck=torch.load(path,map_location=device,weights_only=False)
    cfg=Config(**ck["config"]).validate()
    model=Baby(cfg).to(device);model.load_state_dict(ck["model"]);model.eval()
    return model,cfg,ck

def reuse_random_experience(source,destination):
    """Reuse immutable paired recordings; Replay still enforces causal episode limits."""
    source,destination=Path(source),Path(destination)
    if not source.exists():return 0
    count=0
    for original in source.rglob('*'):
        if not original.is_file():continue
        target=destination/original.relative_to(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            try:os.link(original,target)
            except OSError:shutil.copy2(original,target)
            count+=1
    return count

def adapt_from_checkpoint(root,checkpoint,cfg,protocol,domain,seed,device,deadline,repo,run_id):
    """Each developmental checkpoint starts a separate clone, optimizer and replay.

    Adaptation data seeds depend on training seed and domain, NOT model age.
    Tests use a separate namespace and are never added to replay.
    """
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    model,_,origin=load_model(checkpoint,device)
    seed_all(derived_seed(seed,domain,"adapt-optimizer"))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    replay=Replay(cfg,root/"episodes",derived_seed(seed,domain,"adapt-replay"))
    last=root/"last.pt";ep=0;update=0
    if last.exists():
        ck=torch.load(last,map_location=device,weights_only=False)
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        ep,update=ck["next_block"],ck["update"]
        replay.rng.bit_generator.state=ck["replay_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu());np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
        if torch.cuda.is_available() and ck["cuda_rng"] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        trace=root/"training_trace.jsonl"
        if trace.exists():
            lines=[json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
            trace.write_text("".join(json.dumps(x)+"\n" for x in lines if x["update"]<=update))
    speech=Speech(root/"speech_cache",cfg.sample_rate)
    if not last.exists():save_checkpoint(last,model,optimizer,cfg,0,0,0,replay)
    test_cfg=replace(cfg,physics_trials=protocol["adaptation_trials_per_task_domain"],eval_namespace=protocol["adaptation_namespace"])
    def save_adaptation_state(budget,status):
        pack_paths(root,[root/"episodes"] if (root/"episodes").exists() else [],root/"experience.zip")
        atomic_json(root/"state.json",dict(status=status,budget=budget,updates=update,
                    source_checkpoint_update=origin.get("update"),seed=seed,domain=domain))
        push(root,repo,run_id,["state.json","last.pt","experience.zip","metrics/**","training_trace.jsonl"])
    for budget in protocol["adaptation_budgets"]:
        target_ep=budget//cfg.episode_steps
        expected_update=target_ep*protocol["adaptation_updates_per_episode"]
        metric=root/"metrics"/f"budget_{budget:06d}"/f"eval_{expected_update:08d}.json"
        if ep>target_ep:
            if not metric.exists():raise ValueError("Earlier adaptation evaluation missing; cannot substitute later weights")
            continue
        while ep<target_ep:
            if time.monotonic()>=deadline:
                save_adaptation_state(ep*cfg.episode_steps,"time_budget_reached");return False
            folder=root/"episodes"/f"{ep:07d}"
            collect_episode(cfg,speech,folder,derived_seed(seed,domain,"adapt-train",ep),domain)
            replay.refresh(episode_limit=ep+1);model.train()
            for _ in range(protocol["adaptation_updates_per_episode"]):
                batch,samples=replay.sample(device)
                optimizer.zero_grad(set_to_none=True);loss,stats=losses(model,batch)
                if not torch.isfinite(loss):raise FloatingPointError("Adaptation loss is not finite")
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();update+=1
                with open(root/"training_trace.jsonl","a") as f:f.write(json.dumps(dict(update=update,episode=ep,samples=samples,**stats))+"\n")
            ep+=1
            save_checkpoint(last,model,optimizer,cfg,ep,update,ep*cfg.episode_steps,replay)
        if not metric.exists():
            if time.monotonic()>=deadline:
                save_adaptation_state(ep*cfg.episode_steps,"time_budget_reached");return False
            physical_evaluate(model,test_cfg,speech,device,metric.parent,update,budget,domains=(domain,))
        # Raw adaptation experiences are preserved before local cleanup.
        save_adaptation_state(budget,"completed" if budget==protocol["adaptation_budgets"][-1] else "running")
    return True

def run_suite(protocol_path,output,device,repo=None,run_id="formal-local",resume=False):
    protocol_path=Path(protocol_path).resolve();p=read_json(protocol_path)
    cfg=Config.load(protocol_path.parent/p["training_config"])
    if cfg.evaluation_mode!="physics":raise ValueError("Formal primary endpoint must be physical success")
    if len(set(p["seeds"]))!=len(p["seeds"]):raise ValueError("Duplicate seeds")
    if p["final_namespace"] in {cfg.eval_namespace,p["adaptation_namespace"]}:raise ValueError("Evaluation splits overlap")
    if any(b%cfg.episode_steps for b in p["adaptation_budgets"]):raise ValueError("Adaptation budgets must contain whole episodes")
    if p["adaptation_budgets"]!=sorted(set(p["adaptation_budgets"])) or p["adaptation_budgets"][0]!=0:raise ValueError("Invalid adaptation budgets")
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if resume and repo and not (root/"protocol.json").exists():restore_summary(root,repo,run_id)
    locked={"protocol":p,"training_config":asdict(cfg),"source_archive":__import__('os').environ.get("DEVBABY_CODE_PATH","local")}
    digest=hashlib.sha256(json.dumps(locked,sort_keys=True).encode()).hexdigest()
    locked["sha256"]=digest
    if (root/"protocol.json").exists():
        if read_json(root/"protocol.json")!=locked:raise ValueError("Protocol/source changed. Use a new experiment id.")
        if not resume:raise ValueError("Experiment exists. Use --resume.")
    else:atomic_json(root/"protocol.json",locked)
    started=time.monotonic();deadline=started+p["job_seconds"]
    torch.set_num_threads(cfg.threads)
    state={"status":"running","phase":"training","protocol_sha256":digest,"completed_members":[]}
    def save_state():
        atomic_json(root/"suite_state.json",state)
        push(root,repo,run_id,["protocol.json","suite_state.json","failure.txt","report.html","summary*","*.csv","*.png"])
    def pause():
        state.update(status="time_budget_reached",elapsed_this_job_seconds=time.monotonic()-started)
        save_state();return state
    save_state()
    try:
        # Pair seed and physical experiences; counterbalance variant run order.
        for seed in p["seeds"]:
            variants=list(p["variants"])
            if seed%2:variants.reverse()
            for variant in variants:
                name=f"{variant}-s{seed}";member=root/"members"/name
                member_cfg=replace(cfg,seed=seed,**p["variants"][variant])
                status=read_json(member/"status.json") if (member/"status.json").exists() else {}
                if status.get("status")!="completed":
                    if time.monotonic()>=deadline:return pause()
                    for other_variant in variants:
                        if other_variant==variant:continue
                        other=root/"members"/f"{other_variant}-s{seed}"
                        if not (other/"status.json").exists() or read_json(other/"status.json").get("status")!="completed":continue
                        other_cfg=Config.load(other/"config.json")
                        left=asdict(member_cfg);right=asdict(other_cfg)
                        left.pop('ablation');right.pop('ablation')
                        if left==right and member_cfg.exploration=='random':
                            reused=reuse_random_experience(other/"data/episodes",member/"data/episodes")
                            if reused:print(f'Reusing paired immutable random recordings: {name}',flush=True)
                    old=(member/"config.json").exists()
                    result=train(member_cfg,member,device,repo,f"{run_id}/members/{name}",resume=old,deadline=deadline)
                    if result["status"]!="completed":return pause()
                else:
                    if status.get("updates")!=cfg.blocks*cfg.updates_per_block:raise ValueError("Incomplete member marked completed")
                state["completed_members"].append(name);save_state()
                gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
            if repo:
                # Reuse the recordings across the paired models before freeing scratch.
                # Both result repositories' paths already contain all raw recordings.
                for variant in variants:
                    member=root/"members"/f"{variant}-s{seed}"
                    shutil.rmtree(member/"data",ignore_errors=True)
                    shutil.rmtree(member/"speech_cache",ignore_errors=True)
        # The sealed final panel is first evaluated after ALL base models finish.
        state["phase"]="final_zero_shot";save_state()
        final_cfg=replace(cfg,physics_trials=p["final_trials_per_task_domain"],eval_namespace=p["final_namespace"])
        speech=Speech(root/"evaluation_speech_cache",cfg.sample_rate)
        for seed in p["seeds"]:
            for variant in p["variants"]:
                name=f"{variant}-s{seed}";member=root/"members"/name
                for age in ([0,cfg.blocks] if variant=="full" else [cfg.blocks]):
                    step=age*cfg.updates_per_block
                    destination=member/"final"/f"age_{age:04d}"
                    if (destination/f"eval_{step:08d}.json").exists():continue
                    if time.monotonic()>=deadline:return pause()
                    path=ensure_file(root,f"members/{name}/checkpoints/model_{step:08d}.pt",repo,run_id)
                    model,_,_=load_model(path,device)
                    physical_evaluate(model,final_cfg,speech,device,destination,step,age*cfg.episodes_per_block*cfg.episode_steps)
                    push(member,repo,f"{run_id}/members/{name}",["final/**"])
                    del model
        # Random and privileged reference controllers are simulated once, then reused.
        for policy in ["random","oracle_reference"]:
            destination=root/"baselines"/policy
            if not (destination/"eval_00000000.json").exists():
                if time.monotonic()>=deadline:return pause()
                seed_all(42);model=Baby(cfg).to(device)
                physical_evaluate(model,final_cfg,speech,device,destination,0,0,policy=policy)
                push(root,repo,run_id,["baselines/**"]);del model
        state["phase"]="adaptation";save_state()
        for seed in p["seeds"]:
            for age in p["adaptation_checkpoint_blocks"]:
                step=age*cfg.updates_per_block
                checkpoint=ensure_file(root,f"members/full-s{seed}/checkpoints/model_{step:08d}.pt",repo,run_id)
                for domain in p["adaptation_domains"]:
                    name=f"s{seed}-age{age}-{domain}";destination=root/"adaptation"/name
                    if (destination/"state.json").exists() and read_json(destination/"state.json").get("status")=="completed":continue
                    if time.monotonic()>=deadline:return pause()
                    if repo and resume and (destination/"state.json").exists() and not (destination/"last.pt").exists():
                        archive=ensure_file(root,f"adaptation/{name}/experience.zip",repo,run_id)
                        ensure_file(root,f"adaptation/{name}/last.pt",repo,run_id)
                        import zipfile
                        from .utils import safe_extract
                        with zipfile.ZipFile(archive) as z:safe_extract(z,destination)
                    done=adapt_from_checkpoint(destination,checkpoint,replace(cfg,seed=seed),p,domain,seed,device,deadline,repo,f"{run_id}/adaptation/{name}")
                    if not done:return pause()
                    if repo:
                        shutil.rmtree(destination/"episodes",ignore_errors=True)
                        (destination/"experience.zip").unlink(missing_ok=True)
                        shutil.rmtree(destination/"speech_cache",ignore_errors=True)
                    gc.collect()
        state.update(status="completed",phase="analysis",elapsed_this_job_seconds=time.monotonic()-started)
        atomic_json(root/"suite_state.json",state)
        from .suite_analysis import analyze
        analyze(root)
        state['phase']='completed'
        save_state();print(json.dumps(state),flush=True);return state
    except Exception:
        (root/"failure.txt").write_text(traceback.format_exc())
        state.update(status="failed");save_state();raise

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default="formal_protocol.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--hf-repo")
    parser.add_argument("--run-id",default="formal-local")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    run_suite(args.protocol,args.output,device_for(args.device),args.hf_repo,args.run_id,args.resume)

if __name__=="__main__":main()
