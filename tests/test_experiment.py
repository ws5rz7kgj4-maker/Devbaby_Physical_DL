from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from devbaby.audio import Speech,features
from devbaby.config import Config
from devbaby.world import World,HELDOUT,TRAIN_COMBOS,ACTION_DIM
from devbaby.data import collect_episode,Replay,OBS_KEYS
from devbaby.model import Baby,losses
from devbaby.run import save_checkpoint

@pytest.fixture
def cfg():
    torch.set_num_threads(2)
    return Config(episode_steps=24,clip_steps=8,batch_size=4,threads=2)

def test_speech_is_audible_formant_synthesis(tmp_path):
    speech=Speech(tmp_path)
    a=speech.synthesize('look at the red ball','en-us')
    b=speech.synthesize('look at the blue block','en-gb-scotland')
    assert len(a)>4000 and np.std(a)>.005
    assert len(b)>4000 and np.std(b)>.005
    assert features(a[:2000]).shape==(32,16)
    assert np.array_equal(a,speech.synthesize('look at the red ball','en-us'))

def test_train_holdout_and_true_physics(cfg,tmp_path):
    assert not set(TRAIN_COMBOS)&HELDOUT
    speech=Speech(tmp_path)
    with World(cfg,speech,37,caregiver=False) as w:
        assert not {(x['color'],x['shape']) for x in w.objects}&HELDOUT
        a=w.observe()['proprio'][:3].copy()
        for _ in range(4):o,_,ann=w.step(np.array([1,0,0,0,0],np.float32))
        assert o['proprio'][0]>a[0]+.05
        assert np.isfinite(o['proprio']).all()
        assert all('mass' in obj and 'orientation_xyzw' in obj for obj in ann['objects'])

def test_scene_action_replay_determinism(cfg,tmp_path):
    speech=Speech(tmp_path)
    records=[]
    for _ in range(2):
        with World(cfg,speech,12,caregiver=False) as w:
            frames=[]
            for t in range(5):
                o,_,_=w.step(np.array([.2,-.3,.1,0,.1],np.float32));frames.append(o['rgb'])
            records.append(np.stack(frames))
    assert np.array_equal(*records)

def test_recording_and_annotation_isolation(cfg,tmp_path):
    speech=Speech(tmp_path/'speech');data=tmp_path/'episodes'
    for i in range(2):collect_episode(cfg,speech,data/f'{i:03d}',i+20)
    p=data/'000'
    with np.load(p/'observations.npz') as f:
        assert set(f.files)==OBS_KEYS
        assert f['audio'].shape[0]==f['rgb'].shape[0]==len(f['actions'])+1
        assert np.count_nonzero(f['audio'][0])==0
    scene=json.loads((p/'annotations/scene_and_language.json').read_text())
    assert scene['static_geometry'] and scene['effector']['mass']==.1
    lookup={o['id']:o for o in scene['objects']}
    assert scene['utterances']
    for u in scene['utterances']:
        assert u['related'] and lookup[u['intended_referent']]['color']==u['spoken_color']
    a,_=Replay(cfg,data,1).sample('cpu')
    for f in data.glob('*/annotations/*'):f.write_text('THIS FILE MUST NEVER ENTER THE TRAINER')
    b,_=Replay(cfg,data,1).sample('cpu')
    for key in OBS_KEYS:assert torch.equal(a[key],b[key])

def fake_batch(cfg):
    b,t=cfg.batch_size,cfg.clip_steps
    return {'rgb':torch.rand(b,t+1,3,64,64),'audio':torch.rand(b,t+1,32,16),
            'proprio':torch.rand(b,t+1,10),'actions':torch.rand(b,t,5)*2-1}

def test_gradients_and_frozen_evaluation(cfg):
    torch.manual_seed(4);model=Baby(cfg);b=fake_batch(cfg)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    old=model.visual.net[0].weight.detach().clone()
    loss,stats=losses(model,b);loss.backward();opt.step()
    assert np.isfinite(stats['loss']) and not torch.equal(old,model.visual.net[0].weight)
    assert model.hearing.net[0].weight.grad is not None
    params={k:v.clone() for k,v in model.state_dict().items()}
    model.eval();model.grounding_images(b['rgb'][:,0]);model.grounding_audio(b['audio'])
    for k,v in params.items():assert torch.equal(v,model.state_dict()[k])

def test_no_action_ablation_is_action_invariant(cfg):
    model=Baby(replace(cfg,ablation='no_action')).eval();b=fake_batch(cfg)
    a=model(b);b['actions']=-b['actions'];c=model(b)
    assert torch.equal(a['pred_body'],c['pred_body'])
    assert torch.equal(a['pred_rgb'],c['pred_rgb'])

def test_world_memory_ablation(cfg):
    model=Baby(replace(cfg,ablation='no_world_memory')).eval();b=fake_batch(cfg)
    a=model(b)['h'][:,-1].detach()
    b['rgb'][:,:-2]=0;b['audio'][:,:-2]=0;b['proprio'][:,:-2]=0
    c=model(b)['h'][:,-1].detach()
    assert torch.allclose(a,c,atol=1e-6)

def test_extra_training_label_is_rejected(cfg,tmp_path):
    d=tmp_path/'000';d.mkdir()
    np.savez(d/'observations.npz',rgb=np.zeros((2,64,64,3)),audio=np.zeros((2,32,16)),
             proprio=np.zeros((2,10)),actions=np.zeros((1,5)),semantic_answer=np.zeros(1))
    (d/'complete.json').write_text('{}')
    with pytest.raises(ValueError,match='schema'):Replay(cfg,tmp_path,1).read(d/'observations.npz')

def test_optimizer_checkpoint_roundtrip(cfg,tmp_path):
    torch.manual_seed(5);m=Baby(cfg);o=torch.optim.AdamW(m.parameters(),lr=cfg.learning_rate)
    b=fake_batch(cfg);loss,_=losses(m,b);loss.backward();o.step();o.zero_grad()
    replay=Replay(cfg,tmp_path,6)
    save_checkpoint(tmp_path/'last.pt',m,o,cfg,1,1,24,replay)
    loss,_=losses(m,b);loss.backward();o.step()
    reference={k:v.clone() for k,v in m.state_dict().items()}
    ck=torch.load(tmp_path/'last.pt',weights_only=False)
    n=Baby(cfg);no=torch.optim.AdamW(n.parameters(),lr=cfg.learning_rate)
    n.load_state_dict(ck['model']);no.load_state_dict(ck['optimizer'])
    loss,_=losses(n,b);loss.backward();no.step()
    for k,v in reference.items():assert torch.equal(v,n.state_dict()[k])

def test_progress_explorer_uses_observations_only(cfg,tmp_path):
    from devbaby.data import ProgressExplorer
    model=Baby(replace(cfg,exploration='learning_progress')).eval()
    policy=ProgressExplorer(model,'cpu',4)
    obs={'rgb':np.zeros((64,64,3),np.uint8),'audio':np.zeros((32,16),np.float32),'proprio':np.zeros(10,np.float32)}
    action=policy(obs)
    assert action.shape==(5,) and np.isfinite(action).all()
    batch=fake_batch(cfg)
    loss,stats=losses(model,batch,Baby(cfg).eval())
    assert torch.isfinite(loss) and 'progress_prediction' in stats

def test_iid_candidates_never_contain_holdout_combinations():
    from devbaby.evaluate import candidate_combos
    rng=np.random.default_rng(6)
    for target in TRAIN_COMBOS:
        choices,correct=candidate_combos(target,rng,set(TRAIN_COMBOS))
        assert choices[correct]==target and len(set(choices))==4
        assert not set(choices)&HELDOUT

def test_evaluation_distribution_independent_of_training_q(tmp_path):
    from devbaby.evaluate import prepare_evaluation
    cfg=Config(episode_steps=12,clip_steps=8,eval_per_condition=1,eval_dynamics_episodes=1,probe_examples=3)
    speech=Speech(tmp_path/'speech')
    prepare_evaluation(cfg,speech,tmp_path/'q1')
    prepare_evaluation(replace(cfg,correspondence=0,caregiver_delay_steps=12),speech,tmp_path/'q0')
    for p in (tmp_path/'q1').rglob('*.npz'):
        other=tmp_path/'q0'/p.relative_to(tmp_path/'q1')
        with np.load(p) as a,np.load(other) as b:
            for k in a.files:assert np.array_equal(a[k],b[k]),(p,k)
