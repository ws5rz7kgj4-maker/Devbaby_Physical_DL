"""Verify checkpoint comparison and gradient/optimizer/save paths without retraining."""
import io,json,os,time,zipfile,hashlib,inspect
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import HfApi,hf_hub_download

def main():
    torch.set_num_threads(2);api=HfApi();out=Path('output');out.mkdir(exist_ok=True)
    src='swenlio/devbaby-worldmodel-results';dest='swenlio/devbaby-control-audit-results'
    rev=api.repo_info(src,repo_type='dataset').sha;drev=api.repo_info(dest,repo_type='dataset').sha
    prefix=os.environ['PARAMETER_PREFIX'];hashes={}
    def fetch(repo,path,revision):
        p=Path(hf_hub_download(repo,path,repo_type='dataset',revision=revision))
        hashes[path]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    def publish(phase,result):
        (out/'result.json').write_text(json.dumps(dict(phase=phase,source_revision=rev,data_revision=drev,hashes=hashes,**result),indent=2))
        api.upload_folder(repo_id=dest,repo_type='dataset',folder_path=str(out),path_in_repo=prefix,commit_message='Parameter audit '+phase)
        print('PARAMETER_AUDIT',phase,flush=True)
    publish('started',{})
    source=fetch(dest,'baseline-checks/4fc555c1bed2/source.zip',drev)
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            if name.startswith('devbaby/') and name.endswith('.py'):
                assert '..' not in Path(name).parts;z.extract(name,'.')
    from devbaby.wm_model import WorldModel
    from devbaby.wm_config import WMConfig
    from devbaby.wm_data import Replay
    from devbaby.wm_numeric import clip_finite_gradients
    cfg=WMConfig(batch_size=2,amp=False);checks=[]
    for seed in (0,1):
        ck={}
        for age in (0,65536):
            p=fetch(src,f'runs/worldmodel-dacc6ff327cf/checkpoints/wm-s{seed}-full-age{age}.pt',rev)
            ck[age]=torch.load(p,map_location='cpu',weights_only=False)
        a,b=ck[0]['model'],ck[65536]['model'];assert set(a)==set(b)
        model=WorldModel(cfg);model.load_state_dict(b,strict=True)
        named=list(model.named_parameters());optids=[i for g in ck[65536]['optimizer']['param_groups'] for i in g['params']]
        assert len(named)==len(optids)
        rows=[]
        for (name,p),oid in zip(named,optids):
            delta=(b[name].double()-a[name].double());state=ck[65536]['optimizer']['state'].get(oid,{})
            rows.append(dict(name=name,shape=list(p.shape),requires_grad=p.requires_grad,
                exact_equal=torch.equal(a[name],b[name]),max_abs_change=float(delta.abs().max()),
                changed_elements=int(torch.count_nonzero(delta)),
                old_optimizer_has_state=bool(state),old_optimizer_step=float(state['step']) if 'step' in state else None,
                old_exp_avg_norm=float(state['exp_avg'].double().norm()) if 'exp_avg' in state else None))
        groups={}
        for group in ('encoder','dynamics','filter','image_start','image_decoder','body_decoder'):
            rr=[r for r in rows if r['name'].startswith(group+'.')]
            assert len(rr)>0,('EMPTY_PREFIX',group)
            groups[group]=dict(tensor_count=len(rr),all_exact_equal=all(r['exact_equal'] for r in rr),max_abs_change=max(r['max_abs_change'] for r in rr))
        checks.append(dict(seed=seed,groups=groups,parameters=rows))
    publish('checkpoints_compared',dict(checkpoints=checks))
    rec=fetch(dest,'audits/frozen-control-v1-ba8f8d332928/new_diagnostic_recordings.zip',drev)
    with zipfile.ZipFile(rec) as z:
        for name in z.namelist():
            if name in ('prediction-0/observations.npz','prediction-1/observations.npz'):
                z.extract(name,'data')
    replay=Replay([Path('data/prediction-0'),Path('data/prediction-1')])
    batch=replay.sample(cfg,'full',np.random.default_rng(1234),'cpu')
    path_checks=[]
    for age in (0,65536):
        torch.manual_seed(99);model=WorldModel(cfg);model.load_state_dict(ck[age]['model'],strict=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-5)
        before={k:v.detach().clone() for k,v in model.named_parameters()}
        loss,parts=model.loss(batch);assert torch.isfinite(loss);loss.backward()
        rows=[];included={id(p) for g in optimizer.param_groups for p in g['params']}
        for name,p in model.named_parameters():
            rows.append(dict(name=name,in_optimizer=id(p) in included,grad_present=p.grad is not None,
                grad_norm=float(p.grad.double().norm()) if p.grad is not None else None,
                grad_finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None))
        norm=clip_finite_gradients(model);assert np.isfinite(norm);optimizer.step()
        for row in rows:row['step_max_abs_change']=float((dict(model.named_parameters())[row['name']]-before[row['name']]).abs().max())
        buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
        restored=WorldModel(cfg);restored.load_state_dict(torch.load(buffer,weights_only=True))
        assert all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
        path_checks.append(dict(seed=1,initial_age=age,loss=float(loss.detach()),global_grad_norm=norm,save_reload_exact=True,parameters=rows))
    result=dict(checkpoints=checks,path_checks=path_checks,torch_version=torch.__version__,device='cpu',
        note='Two seeds checked exactly; fresh optimizer one-step checks in FP32 only. Original GPU/AMP path is not reproduced.',
        model_source=inspect.getsource(WorldModel))
    publish('completed',result)

if __name__=='__main__':main()
