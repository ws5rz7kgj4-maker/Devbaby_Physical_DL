"""Apply validated autocast-cache fix to a DevBaby source directory.

Usage: python apply_gradient_fix.py /path/to/DevBaby_WorldModel_Fixed
Does not launch jobs or modify existing checkpoints/results.
"""
import argparse,ast,hashlib,json
from pathlib import Path

def patch(root):
    path=Path(root)/'devbaby/wm_numeric.py'
    original=path.read_text()
    old="with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):"
    new="with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp, cache_enabled=False):"
    if new in original:
        raise ValueError('Cache fix already present; refusing to apply twice. Check existing guard separately.')
    if original.count(old)!=1:raise ValueError('Unexpected source; refusing ambiguous patch')
    marker="    gradients = [p.grad for p in model.parameters() if p.grad is not None]"
    if original.count(marker)!=1:raise ValueError('Gradient guard insertion point missing or ambiguous')
    guard="""    # Every trainable parameter in the current WM/BC loss is expected to participate.
    # Autocast can cache detached GRU weights during no_grad burn-in. Fail early
    # if a parameter is silently disconnected instead of accepting partial updates.
    missing = [name for name, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    if missing:
        raise RuntimeError('Missing training gradients: ' + ', '.join(missing))
"""
    changed=original.replace(old,new).replace(marker,guard+marker)
    changed=changed.replace("NUMERICS_VERSION = 'same-batch-amp-recovery-v1'",
                            "NUMERICS_VERSION = 'same-batch-amp-recovery-no-cache-v2'")
    ast.parse(changed)
    backup=path.with_suffix('.py.before_gradient_fix')
    if backup.exists():raise ValueError('Backup exists; refusing to overwrite')
    backup.write_text(original);path.write_text(changed)
    print(json.dumps(dict(file=str(path),before_sha256=hashlib.sha256(original.encode()).hexdigest(),
        after_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        warning='Use a new run from random initialization; do not resume compromised training checkpoints.'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();patch(a.root)
