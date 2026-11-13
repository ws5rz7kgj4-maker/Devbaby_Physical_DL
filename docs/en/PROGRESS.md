# Progress and experiment ledger

[中文](../zh/PROGRESS.md) · [Home](../../README.md)

Prepared on 2026-11-11 from saved records; this is not a claim that all experiments ran on that date. Results from distinct model generations remain separate. No new training job was submitted during preparation.

| Stage | Completed work / observation | Status and boundary |
|---|---|---|
| Goal and initial implementation | Fully simulated vision, actions, sound and language labels; focus narrowed to physical generalization and learning speed | Language/consciousness were motivations, not completed capability results |
| Early formal physical study | Eight seeds × two conditions; body prediction and coordinate-goal CEM; success/adaptation curves | Complete; restricted physical-learning evidence |
| Early supplement | Delay sweep, history, replay, low LR, encoder freezing, retention, new pushing task | Complete; full updates deteriorate, history is stronger, feedback remains strong |
| World-model prototype | Image/body prediction, action-conditioned latent dynamics, image-goal planning, audio/transfer/probe design | Implementation and protocol, not evidence of competence |
| First world-model run | Terminated on nonfinite gradients | Failed; raw data retained |
| Numerical recovery revision | Same-batch AMP retries, FP32 fallback, FP64 norms, data import/resume checks | Addressed overflow but still missed absent gradients |
| Large world-model run | Saved fitting seal and checkpoints; prediction diagnostics and partial physical evaluation | Canceled on user instruction; compromised training and incomplete formal evaluation |
| Exploration pilot | 96 episodes, 12,288 steps, 16 privileged reference trials | Complete; behavioral coverage only, no neural training |
| Frozen-control diagnostic | Eight seeds × four ages; new prediction and fixed-candidate ranking | Complete; posthoc audit of compromised models |
| FP32 loss comparison | Two losses × 200 updates; shared initialization/data; four train + four development-validation episodes | Complete; object prediction still worse than persistence |
| CPU parameter audit | Seed-0/1 parameter/state comparisons; CPU step and reload | Complete; identified long-term non-updates |
| GPU cache comparison | FP32 / FP16 cache on / FP16 cache off | Complete; defect reproduced and key GRU single-step updates restored |
| Exact gradient patch | Disable cache; reject missing expected gradients; version and backup handling | Delivered; complete repaired training unverified |
| This GitHub snapshot | Recovered early results, recomputed aggregates/some statistics, bilingual documents, figures, CSVs and provenance | Documentation/evidence preparation, not new training |

## Run identities

| Run / output prefix | Purpose | Latest verified record |
|---|---|---|
| `formal-physical-eef2ff40` | Early formal physical learning | completed; 16 members |
| `supplement-c2b8f789d4b0` | Early supplement | completed; training complete |
| `worldmodel-fd0f484f6879` | Initial world model | Failed on nonfinite gradients |
| `worldmodel-dacc6ff327cf` | Numerical-recovery world model | Fitting seal complete; job later canceled |
| `audits/frozen-control-v1-ba8f8d332928` | Frozen diagnostic | completed; no training |
| `baseline-checks/4fc555c1bed2` | FP32 loss comparison | completed; development only |
| `parameter-audits/b5a388062712` | CPU parameter audit | completed |
| `parameter-audits/gpu-d9b63f7d5fd4` | GPU cache reproduction | completed |

Private repository locations, historical job IDs and source hashes are in [provenance](../../provenance/README.md) and [jobs.json](../../provenance/jobs.json).

## Corrections to earlier interpretations

1. “The whole project has no success curves” is incorrect. Early body-control curves exist; **repaired new-world-model** curves are missing.
2. “Random actions prevented the old world model from learning” is not established. The training defect invalidates that attribution, and the original exploration policy has reasonable pilot coverage relative to the tested alternatives.
3. Lower loss/finite aggregate gradients do not establish correct training. Earlier checks missed `grad=None` and persistently initialized parameters.
4. “Patch validated” refers only to the matched GPU single-step mechanism, not complete GradScaler execution, long training or control competence.
5. The defect is not confined to GRUs: another 18 encoder tensors are affected in the parameter records.
6. More prior training does not automatically imply faster adaptation. Early mature models deteriorate under naive updating; the supplement compares mitigation approaches.
7. Several age checkpoints do not establish a scaling law.

## Cancellations, failed attempts and preservation

After the initial world-model failure, only verified raw training data were imported and learners were reinitialized. The historical package named `Fixed` repairs nonfinite-gradient recovery but **still contains the subsequently identified cache defect**.

The first frozen-audit job failed while publishing its initial status because the old output repository was not writable. Evaluation had not begun. A separate private output repository passed an upload/download preflight before the diagnostic completed. The failed attempt has no results to pool with the successful run.

A previous analysis snapshot of the large world-model run recorded 837/1,762 physical panels and 14,712 scored trials. This documentation pass did not enumerate/recount every panel, so these remain historical progress notes rather than inputs to current aggregates or formal inference. The 1,762 panels are planned, not completed. A `state.json` can remain marked running after external cancellation and does not by itself establish live execution.

Synthetic data and checkpoints are preserved for audit. Repaired results need a new run identity, without overwriting the failure evidence or presenting continuation from compromised weights as learning from scratch.
