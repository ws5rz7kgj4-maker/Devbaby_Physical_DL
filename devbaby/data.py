from pathlib import Path
from collections import OrderedDict
import json
import numpy as np
from scipy.io import wavfile
import torch
from .world import World,ACTION_DIM
from .utils import atomic_json,derived_seed

OBS_KEYS={"rgb","audio","proprio","actions"}

class RandomExplorer:
    def __init__(self,seed):
        self.rng=np.random.default_rng(seed)
        self.target=None
        self.remaining=0
        self.grip=0.
    def __call__(self,obs):
        if self.remaining<=0:
            # Targets sampled independently of object identity/position.
            self.target=self.rng.uniform([.05,-.45,.04],[.95,.45,.4])
            self.remaining=int(self.rng.integers(3,12))
            self.grip=float(self.rng.choice([-1,0,1]))
        self.remaining-=1
        d=(self.target-obs["proprio"][:3])/.09
        return np.asarray([*np.clip(d+self.rng.normal(0,.15,3),-1,1),self.grip,self.rng.uniform(-.5,.5)],np.float32)

class ProgressExplorer:
    def __init__(self,model,device,seed):
        self.model,self.device=model,device
        self.random=RandomExplorer(seed)
        self.rng=np.random.default_rng(derived_seed(seed,"choices"))
        self.state=None
        self.prev=np.zeros(ACTION_DIM,np.float32)
    def __call__(self,obs):
        with torch.no_grad():
            h,self.state=self.model.observe(obs,self.prev,self.device,self.state)
            candidates=self.rng.uniform(-1,1,(32,ACTION_DIM)).astype(np.float32)
            candidates[:,4]*=.4
            candidates[0]=self.random(obs)
            if self.rng.random()<.35:
                a=candidates[0]
            else:
                ca=torch.tensor(candidates,device=self.device)
                score=self.model.progress(torch.cat([h.repeat(len(ca),1),self.model._actions(ca)],-1)).squeeze(-1)
                a=candidates[int(score.argmax())]
        self.prev=a
        return a

def collect_episode(cfg,speech,directory,seed,condition="train",model=None,device=None):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    npz=directory/"observations.npz"
    if npz.exists() and (directory/"complete.json").exists():
        return json.loads((directory/"complete.json").read_text())
    policy=ProgressExplorer(model,device,derived_seed(seed,"policy")) if model is not None and cfg.exploration=="learning_progress" else RandomExplorer(derived_seed(seed,"policy"))
    with World(cfg,speech,seed,condition) as world:
        obs=world.observe()
        observations=[obs]
        annotations=[world.annotation()]
        actions=[]
        audio=[]
        for _ in range(cfg.episode_steps):
            action=policy(obs)
            obs,wav,ann=world.step(action)
            observations.append(obs)
            actions.append(action)
            audio.append(wav)
            annotations.append(ann)
        metadata=world.metadata()
        metadata["utterances"]=world.utterances
    arrays={k:np.stack([o[k] for o in observations]) for k in ["rgb","audio","proprio"]}
    arrays["actions"]=np.stack(actions)
    # A strict allow-list; all simulator state goes in a separate directory.
    assert set(arrays)==OBS_KEYS
    with open(directory/"observations.tmp","wb") as f: np.savez_compressed(f,**arrays)
    (directory/"observations.tmp").replace(npz)
    pcm=np.clip(np.concatenate(audio)*32767,-32768,32767).astype(np.int16)
    wavfile.write(directory/"audio.wav",cfg.sample_rate,pcm)
    hidden=directory/"annotations"
    hidden.mkdir(exist_ok=True)
    atomic_json(hidden/"scene_and_language.json",metadata)
    with open(hidden/"frames.jsonl","w",encoding="utf-8") as f:
        for ann in annotations: f.write(json.dumps(ann,ensure_ascii=False)+"\n")
    summary=dict(seed=seed,condition=condition,steps=len(actions),utterances=len(metadata["utterances"]),
                 impacts=sum(e["type"]=="impact" for a in annotations for e in a["events"]),
                 grasps=sum(e["type"]=="grasp" for a in annotations for e in a["events"]),
                 visible_object_frames=sum(o["visible_pixels"]>0 for a in annotations for o in a["objects"]),
                 spoken_concepts=[(u["spoken_color"],u["spoken_shape"]) for u in metadata["utterances"]])
    atomic_json(directory/"complete.json",summary)
    return summary

class Replay:
    """Training loader: cannot load annotation files, transcripts or goal labels."""
    def __init__(self,cfg,root,seed):
        self.cfg,self.root=cfg,Path(root)
        self.rng=np.random.default_rng(seed)
        self.cache=OrderedDict()
        self.refresh()
    def refresh(self,episode_limit=None):
        paths=[p for p in self.root.glob("*/observations.npz") if (p.parent/"complete.json").exists()]
        if episode_limit is not None:
            paths=[p for p in paths if int(p.parent.name)<episode_limit]
        self.paths=sorted(paths)[-self.cfg.replay_episodes:]
    def read(self,path):
        key=str(path)
        if key not in self.cache:
            with np.load(path,allow_pickle=False) as f:
                if set(f.files)!=OBS_KEYS: raise ValueError("Observation schema contains extra or missing keys")
                self.cache[key]={k:f[k] for k in OBS_KEYS}
            if len(self.cache)>min(512,self.cfg.replay_episodes): self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]
    def sample(self,device):
        if not self.paths: raise RuntimeError("No training episodes")
        pieces={k:[] for k in OBS_KEYS}
        trace=[]
        for _ in range(self.cfg.batch_size):
            path=self.paths[int(self.rng.integers(len(self.paths)))]
            ep=self.read(path)
            start=int(self.rng.integers(len(ep["actions"])-self.cfg.clip_steps+1))
            trace.append({"episode":path.parent.name,"start":start})
            for k in OBS_KEYS:
                end=start+self.cfg.clip_steps+(k!="actions")
                pieces[k].append(ep[k][start:end])
        batch={k:torch.as_tensor(np.stack(v),device=device).float() for k,v in pieces.items()}
        batch["rgb"]=batch["rgb"].permute(0,1,4,2,3)/255
        if self.cfg.ablation=="shuffle_audio":
            order=self.rng.permutation(self.cfg.batch_size)
            batch["audio"]=batch["audio"][torch.as_tensor(order,device=device)]
            trace.append({"audio_batch_permutation":order.tolist()})
        return batch,trace

def episode_batch(path,device,start=0,steps=None):
    with np.load(path,allow_pickle=False) as f:
        steps=steps or len(f["actions"])
        b={k:torch.as_tensor(f[k][start:start+steps+(k!="actions")],device=device).float().unsqueeze(0) for k in OBS_KEYS}
    b["rgb"]=b["rgb"].permute(0,1,4,2,3)/255
    return b
