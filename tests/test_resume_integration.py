"""A real interrupted-training continuation test, with identical recorded inputs."""
import shutil
import torch
import pytest
from devbaby.config import Config
from devbaby.run import train
import devbaby.run as runner

def test_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    cfg=Config(episode_steps=12,clip_steps=8,batch_size=4,blocks=2,episodes_per_block=2,
               updates_per_block=2,eval_per_condition=2,eval_dynamics_episodes=1,motor_trials=1,
               motor_steps=3,probe_examples=12,threads=2,save_video=False,max_seconds=600)
    clean=tmp_path/'clean';resumed=tmp_path/'resumed'
    train(cfg,clean,torch.device('cpu'))
    # Reuse identical physical/acoustic inputs, as a checkpoint replay should.
    # Future recorded episodes must not enter replay before their collection block.
    for name in ['data/episodes','evaluation','speech_cache']:
        shutil.copytree(clean/name,resumed/name)
    original=runner.collect_episode
    def crash(*args,**kwargs):
        if str(args[2]).endswith('0000002'):raise RuntimeError('intentional block interruption')
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,'collect_episode',crash)
    with pytest.raises(RuntimeError,match='intentional'):train(cfg,resumed,torch.device('cpu'))
    monkeypatch.setattr(runner,'collect_episode',original)
    train(cfg,resumed,torch.device('cpu'),resume=True)
    a=torch.load(clean/'checkpoints/last.pt',weights_only=False)
    b=torch.load(resumed/'checkpoints/last.pt',weights_only=False)
    assert a['update']==b['update']==4
    assert a['interactions']==b['interactions']==48
    for k,v in a['model'].items():assert torch.equal(v,b['model'][k]),k
    assert (clean/'training_trace.jsonl').read_text()==(resumed/'training_trace.jsonl').read_text()
