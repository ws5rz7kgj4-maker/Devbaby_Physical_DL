import argparse
import copy
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
import traceback
import numpy as np
import torch
from .audio import Speech
from .config import Config
from .data import Replay,collect_episode
from .evaluate import prepare_evaluation,evaluate_all
from .physical import physical_evaluate
from .model import Baby,losses
from .report import write_report,make_video
from .storage import pack_paths,sync,restore
from .utils import atomic_json,derived_seed,device_for,seed_all

class RunConflictError(RuntimeError):
    pass

def save_checkpoint(path,model,opt,cfg,next_block,update,interactions,replay):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(model=model.state_dict(),optimizer=opt.state_dict(),config=asdict(cfg),next_block=next_block,
               update=update,interactions=interactions,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
               python_rng=random.getstate(),replay_rng=replay.rng.bit_generator.state,
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    tmp=path.with_suffix(".tmp");torch.save(value,tmp);tmp.replace(path)

def train(cfg,out,device,hf_repo=None,run_id="local",resume=False,evaluation_root=None,deadline=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if resume and hf_repo and not (out/"checkpoints/last.pt").exists():restore(out,hf_repo,run_id)
    if (out/"config.json").exists():
        existing=Config.load(out/"config.json")
        if asdict(existing)!=asdict(cfg):raise RunConflictError("Output has a different configuration. Use another output/run id.")
        if not resume:raise RunConflictError("Run exists. Use --resume or a new output directory.")
    cfg.save(out/"config.json")
    seed_all(cfg.seed);torch.set_num_threads(cfg.threads)
    environment={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"device":str(device),
                 "source_archive":os.environ.get("DEVBABY_CODE_PATH","local"),"source_revision":os.environ.get("DEVBABY_REVISION","local"),
                 "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                 "dependencies":{p:importlib.metadata.version(p) for p in ["numpy","scipy","pybullet","espeakng-loader","huggingface_hub"]},
                 "pretrained_weights":False,"training_annotation_access":False,"synthetic_data_only":True}
    atomic_json(out/"environment.json",environment)
    print(json.dumps(environment),flush=True)
    speech=Speech(out/"speech_cache",cfg.sample_rate)
    speech.synthesize("look at the red ball",voice="en-us")
    model=Baby(cfg).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=1e-4)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}",flush=True)
    start_block=0;update=0;interactions=0
    replay=Replay(cfg,out/"data/episodes",derived_seed(cfg.seed,"replay"))
    if resume:
        # Load only checkpoints produced by this project / your own result repo.
        ck=torch.load(out/"checkpoints/last.pt",map_location=device,weights_only=False)
        if ck["config"]!=asdict(cfg):raise RunConflictError("Checkpoint/config mismatch")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        start_block,update,interactions=ck["next_block"],ck["update"],ck["interactions"]
        torch.set_rng_state(ck["torch_rng"].cpu());np.random.set_state(ck["numpy_rng"]);random.setstate(ck["python_rng"])
        replay.rng.bit_generator.state=ck["replay_rng"]
        if torch.cuda.is_available() and ck["cuda_rng"] is not None:torch.cuda.set_rng_state_all([x.cpu() for x in ck["cuda_rng"]])
        # Drop records from incomplete blocks; already saved episodes are reused.
        for name,key,limit in [("training_trace.jsonl","update",update),("exposure.jsonl","episode",start_block*cfg.episodes_per_block-1)]:
            path=out/name
            if path.exists():
                records=[json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                path.write_text("".join(json.dumps(r)+"\n" for r in records if r[key]<=limit))
    started=time.monotonic()
    print("Preparing independent evaluation...",flush=True)
    eval_root=Path(evaluation_root) if evaluation_root is not None else out/"evaluation"
    if cfg.evaluation_mode=="multimodal":
        prepare_evaluation(cfg,speech,eval_root)
        if evaluation_root is None:
            pack_paths(out,[eval_root],out/"evaluation.zip")
        else:
            atomic_json(out/"evaluation_reference.json",dict(namespace=cfg.eval_namespace,shared_suite_evaluation=True))
    def assess():
        if cfg.evaluation_mode=="physics":
            return physical_evaluate(model,cfg,speech,device,out/"metrics",update,interactions)
        return evaluate_all(model,cfg,speech,eval_root,device,out/"metrics",update,interactions)
    if not (out/f"metrics/eval_{update:08d}.json").exists():
        if update==0 or start_block%cfg.eval_interval_blocks==0 or start_block==cfg.blocks:
            assess()
        write_report(out)
    save_checkpoint(out/"checkpoints/last.pt",model,optimizer,cfg,start_block,update,interactions,replay)
    initial_snapshot=out/"checkpoints"/f"model_{update:08d}.pt"
    if not initial_snapshot.exists():
        torch.save({"config":asdict(cfg),"model":model.state_dict(),"interactions":interactions,"update":update},initial_snapshot)
    atomic_json(out/"status.json",dict(status="running",next_block=start_block,updates=update,interactions=interactions))
    sync(out,hf_repo,run_id)
    manifest_path=out/"data_manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"shards":[],"recording":"all training observations, actions, PCM audio and separate annotations"}
    stopped=False
    for block in range(start_block,cfg.blocks):
        if time.monotonic()-started>=cfg.max_seconds or (deadline is not None and time.monotonic()>=deadline):
            stopped=True;break
        print(f"Block {block+1}/{cfg.blocks}: random experience collection",flush=True)
        previous=copy.deepcopy(model).eval()
        for par in previous.parameters():par.requires_grad_(False)
        episode_paths=[]
        model.eval()
        for j in range(cfg.episodes_per_block):
            ep=block*cfg.episodes_per_block+j
            directory=out/"data/episodes"/f"{ep:07d}"
            summary=collect_episode(cfg,speech,directory,derived_seed(cfg.seed,"train",ep),
                                    model=model if cfg.exploration=="learning_progress" else None,device=device)
            episode_paths.append(directory)
            with open(out/"exposure.jsonl","a") as f:f.write(json.dumps({"episode":ep,**summary})+"\n")
            interactions+=cfg.episode_steps
        replay.refresh(episode_limit=(block+1)*cfg.episodes_per_block);model.train()
        for u in range(cfg.updates_per_block):
            b,trace=replay.sample(device)
            optimizer.zero_grad(set_to_none=True)
            loss,stats=losses(model,b,previous if cfg.exploration=="learning_progress" else None)
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite training loss")
            loss.backward();grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True))
            optimizer.step();update+=1
            with open(out/"training_trace.jsonl","a") as f:
                f.write(json.dumps(dict(update=update,block=block,interactions=interactions,grad_norm=grad,samples=trace,**stats))+"\n")
            if (u+1)%50==0 or u+1==cfg.updates_per_block:
                print(f" update={update} loss={stats['loss']:.5f} body={stats['body_prediction']:.5f} av={stats['av_contrast']:.4f}",flush=True)
        del previous
        shard=pack_paths(out,episode_paths,out/"data/shards"/f"block_{block:04d}.zip")
        manifest["shards"]=[r for r in manifest["shards"] if r["file"]!=shard["file"]]+[shard]
        atomic_json(manifest_path,manifest)
        if cfg.save_video and not (out/"preview.mp4").exists():make_video(episode_paths[0],out/"preview.mp4",cfg.control_hz)
        if (block+1)%cfg.eval_interval_blocks==0 or block+1==cfg.blocks:
            result=assess()
            summary=result["physical"]["ood_macro_success"] if cfg.evaluation_mode=="physics" else {k:v["accuracy"] for k,v in result["grounding"].items()}
            print(" frozen evaluation: "+json.dumps(summary),flush=True)
        save_checkpoint(out/"checkpoints/last.pt",model,optimizer,cfg,block+1,update,interactions,replay)
        torch.save({"config":asdict(cfg),"model":model.state_dict(),"interactions":interactions,"update":update},out/"checkpoints"/f"model_{update:08d}.pt")
        write_report(out)
        atomic_json(out/"status.json",dict(status="running",next_block=block+1,updates=update,interactions=interactions))
        sync(out,hf_repo,run_id)
    state=json.loads((out/"status.json").read_text())
    state["status"]="time_budget_reached" if stopped else "completed"
    state["elapsed_seconds"]=time.monotonic()-started
    atomic_json(out/"status.json",state);write_report(out);sync(out,hf_repo,run_id)
    print(json.dumps(state),flush=True)
    return state

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/pilot.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--correspondence",type=float)
    parser.add_argument("--exploration",choices=["random","learning_progress"])
    parser.add_argument("--ablation",choices=["none","no_audio","no_action","no_world_memory","shuffle_audio"])
    parser.add_argument("--caregiver-delay-steps",type=int)
    parser.add_argument("--max-seconds",type=int)
    parser.add_argument("--hf-repo")
    parser.add_argument("--run-id",default="local")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();cfg=Config.load(args.config)
    for k in ["seed","correspondence","exploration","ablation","caregiver_delay_steps","max_seconds"]:
        if getattr(args,k) is not None:setattr(cfg,k,getattr(args,k))
    cfg.validate()
    try:train(cfg,args.output,device_for(args.device),args.hf_repo,args.run_id,args.resume)
    except RunConflictError:
        raise
    except Exception:
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        (out/"failure.txt").write_text(traceback.format_exc())
        atomic_json(out/"status.json",dict(status="failed",exception_type=sys.exc_info()[0].__name__))
        try:sync(out,args.hf_repo,args.run_id)
        except Exception:pass
        raise

if __name__=="__main__":main()
