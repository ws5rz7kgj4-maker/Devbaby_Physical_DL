from argparse import Namespace
from types import SimpleNamespace
import json
import pytest
import hf_launch

class API:
    def __init__(self):self.jobs=[];self.calls=0;self.ambiguous=False
    def whoami(self):return {'name':'testuser'}
    def create_repo(self,**kwargs):pass
    def repo_info(self,**kwargs):return SimpleNamespace(private=True)
    def upload_file(self,**kwargs):return SimpleNamespace(oid='a'*40)
    def inspect_job(self,job_id,**kwargs):return next(x for x in self.jobs if x.id==job_id)
    def list_jobs(self,**kwargs):return self.jobs
    def run_job(self,**kwargs):
        self.calls+=1
        assert kwargs['secrets']=={'HF_TOKEN':'unit-test-secret'}
        assert 'unit-test-secret' not in str(kwargs['env'])+str(kwargs['command'])
        j=SimpleNamespace(id=str(self.calls),url=f'https://example.invalid/jobs/{self.calls}',status=SimpleNamespace(stage='RUNNING'),environment=kwargs['env'])
        self.jobs.append(j)
        if self.ambiguous:raise TimeoutError('simulated response loss after server acceptance')
        return j

@pytest.fixture
def setup(monkeypatch,tmp_path):
    monkeypatch.setattr(hf_launch,'ROOT',tmp_path)
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/smoke.json').write_text('{"max_seconds":1200}')
    (tmp_path/'hf_launch.py').write_text('# fake test source')
    api=API();monkeypatch.setattr(hf_launch,'api_client',lambda:(api,'unit-test-secret'))
    args=Namespace(profile='smoke',variant='grounded',seed=0,flavor='t4-small',timeout=None,run_id='test',code_repo=None,result_repo=None,dry_run=False,resume=False)
    return api,args,tmp_path

def test_idempotent_and_no_saved_secret(setup):
    api,args,root=setup
    hf_launch.submit(args);hf_launch.submit(args)
    assert api.calls==1
    for p in (root/'.launch_records').glob('*.json'):assert 'unit-test-secret' not in p.read_text()

def test_lost_response_does_not_duplicate_paid_job(setup):
    api,args,root=setup;api.ambiguous=True
    with pytest.raises(TimeoutError):hf_launch.submit(args)
    hf_launch.submit(args)
    assert api.calls==1

def test_dry_run_no_api_calls(setup):
    api,args,root=setup;args.dry_run=True
    hf_launch.submit(args)
    assert api.calls==0 and not (root/'.launch_records').exists()

def test_source_allowlist_excludes_local_results_and_credentials(setup):
    api,args,root=setup
    before,sha=hf_launch.source_bytes()
    (root/'secrets.json').write_text('{"private":"secret"}')
    (root/'outputs').mkdir();(root/'outputs/data.json').write_text('secret')
    after,sha2=hf_launch.source_bytes()
    assert sha==sha2 and before==after
