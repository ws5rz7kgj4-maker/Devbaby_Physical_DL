import torch
from torch import nn
import torch.nn.functional as F
from .world import ACTION_DIM, PROPRIO_DIM

class ImageEncoder(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,24,5,2,2),nn.SiLU(),nn.Conv2d(24,48,3,2,1),nn.SiLU(),
                               nn.Conv2d(48,64,3,2,1),nn.SiLU(),nn.Conv2d(64,64,3,2,1),nn.SiLU(),
                               nn.Flatten(),nn.Linear(64*4*4,h),nn.LayerNorm(h))
    def forward(self,x): return self.net(x)

class AudioEncoder(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,16,3,2,1),nn.SiLU(),nn.Conv2d(16,32,3,2,1),nn.SiLU(),
                               nn.Flatten(),nn.Linear(32*8*4,h),nn.LayerNorm(h))
    def forward(self,x): return self.net(x.unsqueeze(1))

class Baby(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        h=cfg.hidden
        self.visual=ImageEncoder(h)
        self.hearing=AudioEncoder(h)
        self.audio_memory=nn.GRU(h,h,batch_first=True)
        self.visual_memory=nn.GRU(h,h,batch_first=True)
        self.memory=nn.GRU(h*2+PROPRIO_DIM+ACTION_DIM,h,batch_first=True)
        self.dynamics=nn.Sequential(nn.Linear(h+ACTION_DIM,h),nn.SiLU(),nn.Linear(h,h))
        self.body=nn.Sequential(nn.Linear(h+ACTION_DIM,h),nn.SiLU(),nn.Linear(h,PROPRIO_DIM))
        self.sound=nn.Linear(h,32)
        self.decoder=nn.Sequential(nn.Linear(h,64*4*4),nn.SiLU(),nn.Unflatten(1,(64,4,4)),
                                   nn.ConvTranspose2d(64,48,4,2,1),nn.SiLU(),
                                   nn.ConvTranspose2d(48,24,4,2,1),nn.SiLU(),
                                   nn.ConvTranspose2d(24,3,4,2,1),nn.Sigmoid())
        self.ground_v=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,64))
        self.ground_a=nn.Sequential(nn.Linear(h,h),nn.SiLU(),nn.Linear(h,64))
        self.progress=nn.Sequential(nn.Linear(h+ACTION_DIM,h),nn.SiLU(),nn.Linear(h,1))

    def _actions(self,a):
        return torch.zeros_like(a) if self.cfg.ablation=="no_action" else a

    def _memory(self,x,state=None):
        if self.cfg.ablation=="no_world_memory":
            b,t,d=x.shape
            y,_=self.memory(x.reshape(b*t,1,d),None)
            return y.reshape(b,t,-1),None
        return self.memory(x,state)

    def encode_audio_sequence(self,a):
        b,t=a.shape[:2]
        if self.cfg.ablation=="no_audio": a=torch.zeros_like(a)
        e=self.hearing(a.reshape(b*t,32,16)).reshape(b,t,-1)
        h,_=self.audio_memory(e)
        # Audio-only recurrence integrates raw acoustic intervals; no word boundaries.
        return h[:,-1],e

    def forward(self,batch):
        rgb,aud,pr,act=batch["rgb"],batch["audio"],batch["proprio"],batch["actions"]
        b,t1=rgb.shape[:2]
        t=t1-1
        v=self.visual(rgb.reshape(b*t1,3,64,64)).reshape(b,t1,-1)
        visual_context,_=self.visual_memory(v[:,:t])
        aseq,e=self.encode_audio_sequence(aud[:,:t])
        prev=torch.cat([torch.zeros_like(act[:,:1]),act[:,:t-1]],dim=1)
        x=torch.cat([v[:,:t],e,pr[:,:t],self._actions(prev)],dim=-1)
        h,_=self._memory(x)
        ha=torch.cat([h,self._actions(act)],dim=-1)
        z=self.dynamics(ha)
        pred_rgb=self.decoder(z.reshape(b*t,-1)).reshape(b,t,3,32,32)
        recon=self.decoder(v[:,:t].reshape(b*t,-1)).reshape(b,t,3,32,32)
        return dict(h=h,z=z,target_z=v[:,1:].detach(),pred_rgb=pred_rgb,recon=recon,
                    pred_body=pr[:,:t]+self.body(ha),pred_audio=self.sound(z),
                    vg=F.normalize(self.ground_v(visual_context[:,-1]),dim=-1),
                    ag=F.normalize(self.ground_a(aseq),dim=-1),progress=self.progress(ha.detach()).squeeze(-1))

    @torch.no_grad()
    def observe(self,obs,previous_action,device,state=None):
        rgb=torch.as_tensor(obs["rgb"],device=device).permute(2,0,1).float().unsqueeze(0)/255
        a=torch.as_tensor(obs["audio"],device=device).float().unsqueeze(0)
        if self.cfg.ablation=="no_audio": a.zero_()
        v=self.visual(rgb)
        e=self.hearing(a)
        pr=torch.as_tensor(obs["proprio"],device=device).float().unsqueeze(0)
        act=torch.as_tensor(previous_action,device=device).float().reshape(1,-1)
        x=torch.cat([v,e,pr,self._actions(act)],-1).unsqueeze(1)
        y,state=self._memory(x,state)
        return y[:,0],state

    @torch.no_grad()
    def grounding_images(self,rgb):
        if rgb.ndim==4:
            v=self.visual(rgb).unsqueeze(1).repeat(1,self.cfg.clip_steps,1)
        else:
            n,t=rgb.shape[:2]
            v=self.visual(rgb.reshape(n*t,3,64,64)).reshape(n,t,-1)
        h,_=self.visual_memory(v)
        return F.normalize(self.ground_v(h[:,-1]),dim=-1)

    @torch.no_grad()
    def grounding_audio(self,a):
        h,_=self.encode_audio_sequence(a)
        return F.normalize(self.ground_a(h),dim=-1)

def losses(model,batch,previous_model=None):
    out=model(batch)
    b,t=out["pred_rgb"].shape[:2]
    target=F.interpolate(batch["rgb"][:,1:].reshape(b*t,3,64,64),size=32,mode="area").reshape(b,t,3,32,32)
    current=F.interpolate(batch["rgb"][:,:t].reshape(b*t,3,64,64),size=32,mode="area").reshape(b,t,3,32,32)
    prediction=F.mse_loss(out["pred_rgb"],target)
    recon=F.mse_loss(out["recon"],current)
    body=F.mse_loss(out["pred_body"],batch["proprio"][:,1:])
    latent=F.mse_loss(out["z"],out["target_z"])
    snd=F.mse_loss(out["pred_audio"],batch["audio"][:,1:].mean(-1))
    if model.cfg.ablation=="no_audio": snd=prediction*0
    # Gate by measured acoustic energy only, never utterance/semantic annotations.
    active=batch["audio"][:,:t].square().mean((1,2,3))>1e-5
    av=prediction*0
    if active.sum()>=2 and model.cfg.ablation!="no_audio":
        vg,ag=out["vg"][active],out["ag"][active]
        score=vg@ag.T/.15
        y=torch.arange(len(vg),device=score.device)
        av=(F.cross_entropy(score,y)+F.cross_entropy(score.T,y))/2
    progress=prediction*0
    if previous_model is not None:
        with torch.no_grad():
            old=previous_model(batch)
            # Stable observable coordinates; latent spaces from different checkpoints are not compared.
            olderr=(old["pred_body"]-batch["proprio"][:,1:]).square().mean(-1)
            newerr=(out["pred_body"].detach()-batch["proprio"][:,1:]).square().mean(-1)
            improvement=(olderr-newerr).clamp(-1,1)
        progress=F.mse_loss(out["progress"],improvement)
    total=prediction+recon+.2*latent+2*body+.1*snd+.25*av+.2*progress
    return total,{"loss":float(total.detach()),"pixel_prediction":float(prediction.detach()),
                  "reconstruction":float(recon.detach()),"body_prediction":float(body.detach()),
                  "latent_prediction":float(latent.detach()),"audio_prediction":float(snd.detach()),
                  "av_contrast":float(av.detach()),"progress_prediction":float(progress.detach())}
