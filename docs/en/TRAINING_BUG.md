# Training defect, evidence and repair boundary

[中文](../zh/TRAINING_BUG.md) · [Home](../../README.md)

## Confirmed problem

The later world model runs `WorldModel.loss` burn-in under `torch.no_grad()` and subsequent trainable forwards inside the same FP16 autocast context. Matched tests reproduce missing gradients caused by cached-weight reuse in this implementation. Other parameters can still update, so decreasing total loss, finite aggregate gradient norms and increasing step counters do not establish that the entire model learns.

This finding concerns the archived implementation and recorded PyTorch 2.6.0 / L4 environment. It is not a claim about every PyTorch version or AMP use. Earlier body-prediction experiments use a different model/training path and are not invalidated solely by this defect.

## Evidence chain

1. Strictly load seed 0 and seed 1 checkpoints at ages 0 and 65,536; each records 4,096 updates.
2. All four `dynamics` and four `filter` parameter tensors are bitwise unchanged and have no optimizer state. Nonempty parameter sets exclude an empty-prefix comparison artifact.
3. Full records additionally show 18 unchanged encoder tensors without optimizer state. Four encoder LayerNorm tensors do change; describing the entire encoder as normally trained would be incorrect.
4. CPU FP32 tests on real synthetic sensory batches produce finite gradients and single-step changes in the key parameters, with exact save/reload agreement.
5. Matched GPU single-step tests reproduce the cache-on/cache-off difference using the same initialization and batch.

| GPU condition | Missing-gradient tensors / 44 | Finite-gradient tensors / 44 | Nonzero single-step changes / 44 | Eight GRU tensors |
|---|---:|---:|---:|---|
| FP32 | 0 | 44 | 44 | All have gradients and update |
| FP16, cache on | 26 | 18 | 16 | All missing gradients; no updates |
| FP16, cache off | 0 | 44 | 42 | All have gradients and update |

Both `image_start` tensors have zero gradients and zero changes in this FP16 single-step test without GradScaler, including cache-off. Thus the evidence does not establish that every parameter updates after disabling the cache. Both tensors did change in the historical long-run checkpoints.

Raw evidence: [CPU](../../results/raw/parameter_cpu_result.json), [GPU](../../results/raw/parameter_gpu_result.json). See the [data dictionary](../../results/README.md) for parameter-level CSVs.

## Patch

The archived [apply_gradient_fix.py](../../code/gradient_audit/apply_gradient_fix.py) applies an exact source edit to the original project:

```python
with torch.autocast(
    device_type=device.type,
    dtype=torch.float16,
    enabled=use_amp,
    cache_enabled=False,
):
    ...
```

Before gradient clipping, it rejects `grad is None` for trainable parameters expected to participate in the loss. Current WM/BC paths expect all trainable parameters; future optional heads or conditional branches must maintain an explicit expected set so valid unused parameters are not falsely rejected.

The script updates the numerics version, checks exact source matches and syntax, saves a backup and refuses repeated application or backup overwriting. It modifies source only and submits no cloud jobs. This repository preserves historical packages rather than relabeling them as a validated repaired training release.

## Validation boundary

| Check | Status |
|---|---|
| Historical seed 0/1 frozen parameters and missing optimizer state | Verified |
| Matched GPU cache-on/cache-off reproduction | Verified |
| Restored key recurrent gradients and single-step updates | Verified |
| Repaired full `optimizer_update` + GradScaler retries/fallback/resume | No completed regression evidence |
| Repaired small-data prediction and simple physical tasks | No completed evidence |
| Repaired full-scale training and generalization curves | No completed evidence |

Earlier overflow recovery tests addressed nonfinite gradients but missed `grad=None`. This is an implementation and acceptance-test gap, not an explanation based on exploration difficulty. The full update path and per-module changes must pass before scaling compute.

## Corrections to historical interpretations

- Old AMP world-model prediction/control metrics remain **descriptive records of a defective implementation**. They do not establish that random exploration is ineffective or support normal-training learning laws, causal audio benefits or scaling laws.
- Simulator trajectories and annotations are not automatically invalidated by an optimizer defect. They may be reused after identity/split checks; compromised learned weights should not initialize repaired capability comparisons.
- The separate 200-update **FP32** loss comparison does not use this AMP-cache path. Its prediction-quality problem remains unresolved by the patch.
- Earlier body-prediction/reaching results require their own assessment; see [early experiments](EARLY_EXPERIMENTS.md).

This GitHub preparation rebuilds documentation, tables and figures only. It launches no new training or physical evaluation.
