from dataclasses import asdict,replace
import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch
from devbaby.config import Config
from devbaby.audio import Speech
from devbaby.world import World
from devbaby.model import Baby
from devbaby.physical import LearnedController,Goal,physical_evaluate
from devbaby.physical_report import curve_statistics,normalized_auc
from devbaby.suite_analysis import sign_flip_p,holm
from devbaby.utils import derived_seed

def tiny_config():
    return Config(seed=0,blocks=2,episodes_per_block=2,episode_steps=12,clip_steps=8,batch_size=4,
                  updates_per_block=2,evaluation_mode='physics',eval_interval_blocks=1,
                  motor_domain_randomization=True,physics_trials=1,physics_steps=4,
                  planner_horizon=2,planner_candidates=8,planner_iterations=1,
                  threads=2,save_video=False,max_seconds=300)

def test_rates_and_censoring():
    s=curve_statistics([0,1000,2000],[.1,.2,.15])
    assert s['growth_pp_per_1000_interactions'][1:]==pytest.approx([10,-5])
    assert s['sustained_threshold_interactions']['0.5'] is None
    assert normalized_auc([0,1000],[0,1])==.5
    assert curve_statistics([0,1000,2000],[.1,.55,.6])['sustained_threshold_interactions']['0.5']==1000

def test_seed_level_statistics():
    assert sign_flip_p([1]*8)==2/256
    assert sign_flip_p([0]*8)==1
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])

def test_higher_start_is_not_faster_learning():
    from devbaby.suite_analysis import adaptation_gain_auc
    assert adaptation_gain_auc([0,1000],[.8,.8])==0
    assert adaptation_gain_auc([0,1000],[.1,.1])==0
    assert adaptation_gain_auc([0,1000],[.1,.3])==pytest.approx(.1)
    assert adaptation_gain_auc([0,1000],[.9,.8])==pytest.approx(-.05)

def test_partial_matrix_cannot_be_reported_complete(tmp_path):
    from devbaby.suite_analysis import analyze
    (tmp_path/'protocol.json').write_text(json.dumps({'protocol':{},'training_config':{}}))
    (tmp_path/'suite_state.json').write_text('{"status":"time_budget_reached"}')
    with pytest.raises(ValueError,match='partial'):analyze(tmp_path)

def test_actuator_holdouts(tmp_path):
    cfg=tiny_config();speech=Speech(tmp_path)
    with World(cfg,speech,12,'train') as a, World(cfg,speech,12,'new_dynamics') as b,World(cfg,speech,12,'new_delay') as c:
        assert .08<=a.hand_mass<=.12 and .20<=b.hand_mass<=.28
        assert a.motor_delay==0 and c.motor_delay in (2,3)
        old=c.target.copy();c.step(np.array([1,0,0,0,0],np.float32))
        assert np.array_equal(c.target,old)
        assert c.metadata()['effector']['command_delay_steps']==c.motor_delay

def test_q_changes_audio_not_physical_experience(tmp_path):
    cfg=replace(tiny_config(),episode_steps=36)
    speech=Speech(tmp_path)
    with World(cfg,speech,61) as a,World(replace(cfg,correspondence=0),speech,61) as b:
        for _ in range(36):
            act=np.array([.1,-.2,.1,1,0],np.float32)
            oa,_,_=a.step(act);ob,_,_=b.step(act)
            assert np.array_equal(oa['rgb'],ob['rgb'])
            assert np.array_equal(oa['proprio'],ob['proprio'])
        assert [u['start_s'] for u in a.utterances]==[u['start_s'] for u in b.utterances]

def test_controller_does_not_use_annotation_fields():
    torch.set_num_threads(2);cfg=tiny_config();torch.manual_seed(1);model=Baby(cfg)
    obs={'rgb':np.zeros((64,64,3),np.uint8),'audio':np.zeros((32,16),np.float32),'proprio':np.zeros(10,np.float32)}
    goal=Goal('reach',(.4,.1,.2),(.3,0,.1))
    a=LearnedController(model,cfg,'cpu',3)(obs,goal)
    obs['object_position_oracle']=[999,999,999];obs['success_label']=True
    b=LearnedController(model,cfg,'cpu',3)(obs,goal)
    assert np.array_equal(a,b)

def test_physical_evaluation_frozen_and_separated(tmp_path):
    torch.set_num_threads(2);cfg=tiny_config();model=Baby(cfg);speech=Speech(tmp_path/'speech')
    before={k:v.clone() for k,v in model.state_dict().items()}
    r=physical_evaluate(model,cfg,speech,'cpu',tmp_path/'result',0,0,domains=('new_dynamics',))
    assert r['physical']['n_trials']==3 and 0<=r['physical']['ood_macro_success']<=1
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert derived_seed('train',1)!=derived_seed('physical-sealed-final-v1',1)

def test_adaptation_resume_preserves_earlier_budgets(tmp_path):
    from devbaby.suite import adapt_from_checkpoint
    cfg=tiny_config();torch.set_num_threads(2);torch.manual_seed(6)
    model=Baby(cfg);source=tmp_path/'base.pt'
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'update':0},source)
    protocol={'adaptation_budgets':[0,12,24],'adaptation_updates_per_episode':1,
              'adaptation_trials_per_task_domain':1,'adaptation_namespace':'unit-adapt-test'}
    out=tmp_path/'adapt';deadline=time.monotonic()+300
    assert adapt_from_checkpoint(out,source,cfg,protocol,'new_delay',0,'cpu',deadline,None,'test')
    files={str(p.relative_to(out)):p.read_bytes() for p in out.glob('metrics/**/eval_*.json')}
    assert adapt_from_checkpoint(out,source,cfg,protocol,'new_delay',0,'cpu',deadline,None,'test')
    assert files=={str(p.relative_to(out)):p.read_bytes() for p in out.glob('metrics/**/eval_*.json')}
    unchanged=torch.load(source,weights_only=False)
    for k,v in model.state_dict().items():assert torch.equal(v,unchanged['model'][k])

def test_adaptation_interruption_matches_fixed_data(tmp_path,monkeypatch):
    import shutil
    import devbaby.suite as suite
    cfg=tiny_config();torch.set_num_threads(2);torch.manual_seed(7)
    source=tmp_path/'base.pt';model=Baby(cfg)
    torch.save({'config':asdict(cfg),'model':model.state_dict(),'update':0},source)
    protocol={'adaptation_budgets':[0,24,36],'adaptation_updates_per_episode':1,
              'adaptation_trials_per_task_domain':1,'adaptation_namespace':'unit-resume-adapt'}
    clean=tmp_path/'clean';resumed=tmp_path/'resumed';deadline=time.monotonic()+300
    suite.adapt_from_checkpoint(clean,source,cfg,protocol,'new_delay',0,'cpu',deadline,None,'test')
    shutil.copytree(clean/'episodes',resumed/'episodes')
    original=suite.collect_episode
    def fail(*args,**kwargs):
        if Path(args[2]).name=='0000001':raise RuntimeError('intentional adaptation interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(suite,'collect_episode',fail)
    with pytest.raises(RuntimeError,match='intentional'):
        suite.adapt_from_checkpoint(resumed,source,cfg,protocol,'new_delay',0,'cpu',deadline,None,'test')
    monkeypatch.setattr(suite,'collect_episode',original)
    suite.adapt_from_checkpoint(resumed,source,cfg,protocol,'new_delay',0,'cpu',deadline,None,'test')
    a=torch.load(clean/'last.pt',weights_only=False);b=torch.load(resumed/'last.pt',weights_only=False)
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    for p in (clean/'metrics').glob('**/eval_*.json'):assert p.read_bytes()==(resumed/'metrics'/p.relative_to(clean/'metrics')).read_bytes()

def test_full_physical_suite_end_to_end(tmp_path):
    from devbaby.suite import run_suite
    cfg=tiny_config();cfg.save(tmp_path/'config.json')
    protocol={'version':'functional-test','development_only':True,'training_config':'config.json',
              'seeds':[0,1],'variants':{'full':{},'no_action':{'ablation':'no_action'}},
              'final_namespace':'unit-final','final_trials_per_task_domain':1,
              'adaptation_domains':['new_delay'],'adaptation_budgets':[0,12],
              'adaptation_checkpoint_blocks':[0,1,2],'adaptation_updates_per_episode':1,
              'adaptation_trials_per_task_domain':1,'adaptation_namespace':'unit-adapt','job_seconds':300,
              'primary_family':['final_minus_initial','final_minus_no_action','adaptation_auc_gain']}
    path=tmp_path/'protocol.json';path.write_text(json.dumps(protocol))
    result=run_suite(path,tmp_path/'suite',torch.device('cpu'))
    assert result['status']=='completed'
    summary=json.loads((tmp_path/'suite/summary.json').read_text())
    assert summary['development_only'] and not any(x['positive_evidence'] for x in summary['primary_tests'])
    assert (tmp_path/'suite/physical_learning.png').exists()
    for seed in [0,1]:
        left=tmp_path/'suite/members'/f'full-s{seed}'/'data/episodes'
        right=tmp_path/'suite/members'/f'no_action-s{seed}'/'data/episodes'
        for file in left.glob('*/observations.npz'):assert file.read_bytes()==(right/file.relative_to(left)).read_bytes()
    assert run_suite(path,tmp_path/'suite',torch.device('cpu'),resume=True)['status']=='completed'
