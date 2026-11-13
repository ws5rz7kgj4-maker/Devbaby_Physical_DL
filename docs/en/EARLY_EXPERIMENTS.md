# Early physical-control experiments and supplemental study

[中文](../zh/EARLY_EXPERIMENTS.md) · [English home](../../README.md)

This document records completed experiments with the **earlier `Baby` body-dynamics predictor**. This system differs from the later `wm_model.WorldModel` implementation with two-view image goals and action-conditioned latent dynamics. The later autocast-cache failure must not be retroactively assigned to the earlier implementation: the recovered earlier training code does not use that AMP burn-in path. This documentation pass checked saved completion states, protocols, CSV files and per-seed final summaries. It did not retrain models or independently re-audit every raw trajectory.

## 1. System and completed scope

Run `formal-physical-eef2ff40` is recorded as `completed`: 8 training seeds × 2 conditions (full model and action-input ablation), or 16 completed members. Each model received 32,768 simulated interactions and 8,192 optimizer updates. The two conditions reuse recorded random exploration within each seed: the sum of per-model available-interaction budgets is 524,288, representing 262,144 unique base-environment transitions under paired reuse. Replay resamples transitions, so this budget sum is not the optimizer's total transition-read count.

Training data came entirely from PyBullet, procedural impact audio and eSpeak NG speech. Models were randomly initialized and fitted using RGB, acoustic features, proprioception and issued actions. No transcripts, speech recognizer, pretrained model, object-coordinate labels or task-success rewards entered model fitting. Speech templates, referents, voices and timing were randomized while retaining programmed associations with scene properties.

**Capability boundary:** this is body-motion control with a simplified point hand and a programmed proximity-grasp primitive. The experiment supplies the CEM planner, task name, spatial target, contact coordinates and task cost. Touch/lift coordinates are derived from simulator state at task initialization. The learned policy does not read live object positions, mass, friction or actuator delay. It does not infer tasks autonomously from speech or goal images, and is not an assumption-free infant model.

| Task | Executed success criterion | Supplied assistance |
|---|---|---|
| reach | Hand within 8 cm of target for the final 3 steps | Spatial coordinates and distance cost; open gripper during planning |
| touch | Actual physical contact for any 3 consecutive steps | Initial contact coordinates and distance cost |
| grasp_lift | For the final 3 steps: selected object held, its center at least 14 cm above its initial height, hand within 13 cm of lift goal | Initial contact point, lift coordinates, grasp-then-lift cost and grasp primitive |

Evaluation lasts 40 control steps. Development panels use 6 scenes per task/domain; the separate sealed final panel uses 24. The OOD macro score weights 4 changed domains × 3 tasks equally, with IID reported separately. Training seeds are independent model repetitions; paired evaluation scenes are not additional model repetitions.

Formal reach/touch/grasp_lift task execution sets `caregiver=False`: caregiver speech is disabled during these evaluations, although physical impact audio can still occur. Training speech exposure must not be confused with live linguistic assistance during task execution.

## 2. Recorded physical-success learning curves

The following values were recomputed from [formal_success_curves.csv](../../results/raw/early/formal_success_curves.csv). Each age aggregates 8 model seeds on 72 paired scene–task–domain cells, giving 576 scored trials, not 576 independent scenes.

| Base interactions | OOD development successes/scored trials | Success | Interval growth: percentage points per 1,000 interactions |
|---:|---:|---:|---:|
| 0 | 2/576 | 0.35% | — |
| 8,192 | 360/576 | 62.50% | 7.59 |
| 16,384 | 380/576 | 65.97% | 0.42 |
| 24,576 | 401/576 | 69.62% | 0.45 |
| 32,768 | 427/576 | 74.13% | 0.55 |

The first 8,192 interactions account for about 84.2% of the net development-curve increase. This supports learning progress for this body-prediction controller under the supplied task interface. Growth is a finite difference, not an instantaneous per-interaction learning rate, an exact emergence time or a scaling law. All 8 seeds first reached two consecutive scheduled checkpoints at or above 50% starting at 8,192 interactions; none reached two consecutive checkpoints at or above 80%. Threshold timing is only localized between scheduled observations.

### Separate sealed final test

The [per-seed final CSV](../../results/raw/early/formal_final_success.csv) agrees with the eight `formal_final_full_s*.json` files: full-model OOD success is **72.92% (1,680/2,304)**, initial-model success **0.74% (17/2,304)**, and no-action success **0.35% (8/2,304)**. The random policy scores 1.39% and the privileged scripted reference 95.14%; these two references each use one shared set of 288 OOD scene–task trials and do not have eight independently trained model seeds.

| Domain | reach | touch | grasp_lift | Three-task mean |
|---|---:|---:|---:|---:|
| New IID scenes | 100.00% | 84.90% | 67.71% | 84.20% |
| New dynamics | 100.00% | 86.98% | 67.19% | 84.72% |
| New layout | 100.00% | 73.96% | 75.00% | 82.99% |
| New action delay | 39.58% | 72.92% | 53.65% | 55.38% |
| Joint physical shifts | 74.48% | 72.92% | 58.33% | 68.58% |

Each task/domain cell contains 8 seeds × 24 scenes = 192 scored trials. The sealed endpoint of 72.92% and development endpoint of 74.13% come from different panels; they must not be connected into one learning curve or interpreted as training regression. Joint shifts are not simply harder than the isolated delay shift in this task set.

### Meaning of “delay” in this version

The earlier base experiment uses zero command-queue delay during training/IID. `new_delay` and `joint_physics` sample **2 or 3 control steps** per scene. At 8 Hz, issued commands start affecting the actuator **250 or 375 ms** later, through a FIFO queue. This is motor-command delay, not delayed language, labels or reward. Even at zero queue delay the hand retains mass, damping and servo dynamics; it does not teleport.

## 3. Further adaptation can reduce mature-model performance

[adaptation_curves.csv](../../results/raw/early/formal_adaptation_curves.csv) compares three base ages, two environments and four additional-experience budgets on a separate adaptation panel.

| Base-interaction age | 0 extra | 128 extra | 512 extra | 2,048 extra |
|---:|---:|---:|---:|---:|
| 0 | 1.39% | 4.86% | 34.72% | 31.25% |
| 16,384 | 57.64% | 53.12% | 49.31% | 50.00% |
| 32,768 | 64.93% | 49.31% | 50.35% | 49.65% |

These means weight 2 environments × 3 tasks equally, using 8 model seeds and 36 fixed scene–task–environment cells. Budgets of 128/512/2,048 interactions correspond to 32/128/512 optimizer updates. The mature model loses **15.28 percentage points** overall: new-dynamics success falls from 77.78% to 69.44%, and new-delay success from 52.08% to 29.86%. All eight seeds decline in the delay environment.

Higher starting performance and ceiling effects complicate gain comparisons across ages, but do not by themselves explain absolute deterioration of the same mature model on the same panel. The supported conclusion is that **this adaptation procedure lowers task performance**. It does not isolate the causal contributions of forgetting, temporal modeling, data coverage or planning error.

### Prespecified primary comparisons

Source: [formal_summary.json](../../results/raw/early/formal_summary.json). The first two rows use percentage points; the third uses a difference in interaction-budget-normalized, baseline-subtracted success-gain AUC.

| Comparison | Mean difference | Paired-seed bootstrap 95% CI | Holm p |
|---|---:|---:|---:|
| Final full − initial: sealed OOD success | +72.1788 pp | [70.8767, 73.6111] pp | 0.023438 |
| Final full − no action: sealed OOD success | +72.5694 pp | [71.0069, 74.3490] pp | 0.023438 |
| Mature − initial: adaptation gain AUC | −0.417752 | [−0.510745, −0.324219] | 0.023438 |

These use exact two-sided sign-flip tests over eight training seeds and Holm correction over three comparisons. Confidence intervals resample model seeds only, conditional on the fixed test panel; they omit scene-sampling uncertainty. They are not precise intervals for arbitrary scene distributions. The third effect is negative and does not support faster learning after maturation.

## 4. Completed supplemental study

The [state](../../results/raw/early/supplement_supplement_state.json) for `supplement-c2b8f789d4b0` records `completed` and `training_complete=true`. The study reuses eight earlier Baby seeds and their experience, and trains a randomly initialized finite-history body-dynamics network, `history`, that explicitly receives the past eight proprioception/issued-action pairs. It matches experience budgets and optimizer-step counts with Baby, but differs in sensors, architecture, parameter count and valid prediction-target count. It is not a single-factor architecture ablation or an exactly compute-matched comparison.

Fresh panels compare frozen weights, full updates at the original learning rate, a 0.1× learning rate, frozen perception encoders, 50% old-experience replay, and the finite-history baseline. Adaptation adds up to 2,048 interactions in `new_dynamics`, fixed `delay_3` and fixed `delay_6`. Final evaluation uses 32 scenes per task/domain; development/validation uses 12, and IID retention uses 12 per task.

The [complete curve CSV](../../results/raw/early/supplement_curves.csv) contains **4,918 evaluation cells and 76,736 logical scored trial rows**. Identical policies/panels can reuse physical trajectories, so the latter must not be described as 76,736 independent newly executed rollouts. This documentation pass did not recount deduplicated physical trajectories.

### Primary endpoint: mature model plus 2,048 additional interactions

Nine task/domain cells are equally weighted. Each method contributes 8 model seeds × 9 cells × 32 scenes = 2,304 scored trials.

| Method | Macro success |
|---|---:|
| `frozen` | 47.87% |
| `full_update` | 38.85% |
| `low_lr` | 40.54% |
| `frozen_encoder` | 40.32% |
| `replay_mix` | 44.79% |
| `history` | 59.29% |

Source: [contrasts.csv](../../results/raw/early/supplement_contrasts.csv). These are prespecified comparisons at the sealed endpoint.

| Comparison | Success difference (pp) | Crossed seed/scene bootstrap 95% CI (pp) | Holm p |
|---|---:|---:|---:|
| Full update − frozen | −9.0278 | [−12.3698, −5.7726] | 0.046875 |
| Replay − full update | +5.9462 | [3.3409, 8.8108] | 0.046875 |
| Frozen encoder − full update | +1.4757 | [−1.2587, 4.2101] | 0.281250 |
| Low LR − full update | +1.6927 | [−1.0851, 4.6875] | 0.281250 |
| History − full update | +20.4427 | [15.8854, 24.8698] | 0.046875 |
| History − frozen | +11.4149 | [6.7708, 16.1024] | 0.046875 |

Within this experiment, full updates reduce success; replay partially mitigates that loss relative to full updates, although its macro mean remains below frozen performance. The finite-history baseline outperforms both principal Baby controls at the endpoint. Freezing perception or reducing the learning rate does not show a clear improvement under the prespecified tests. Eight seeds remain the independent model units, with Holm correction over all six tests; crossed bootstrap intervals preserve paired scenes across methods.

### Delay-strength curves before adaptation

| Delay steps | Seconds | Baby | Finite history | Nonlearned feedback reference |
|---:|---:|---:|---:|---:|
| 0 | 0.000 | 84.90% | 90.89% | 100.00% |
| 1 | 0.125 | 80.86% | 85.29% | 100.00% |
| 2 | 0.250 | 67.32% | 70.83% | 98.96% |
| 3 | 0.375 | 41.54% | 45.83% | 90.63% |
| 4 | 0.500 | 25.78% | 27.08% | 87.50% |
| 6 | 0.750 | 18.88% | 21.61% | 42.71% |

The three tasks are equally weighted. Baby/history each use eight seeds and 32 scenes per task. Feedback is one fixed engineering policy, not eight trained seeds. It uses observable proprioception, explicit goals and hand-written control rules, without reading live hidden physical state. Its strong performance means **these tasks have not established an advantage for the learned system over simple feedback control**. Finite history also fails to eliminate poor performance at large delays.

### New pushing task: substantial distribution-shift failures

Pushing uses a supplied approach-contact-then-transport decomposition; neither model explicitly predicts object displacement. The task and geometry are new within the same PyBullet simulator, not evidence of cross-simulator or arbitrary-task generalization.

| Environment | Baby push success | Finite-history push success |
|---|---:|---:|
| IID | 67.58% | 67.58% |
| New dynamics | 5.86% | 4.30% |
| New layout | 58.98% | 60.16% |
| Fixed 3-step delay | 0.39% | 0.00% |

Each value averages eight seeds × 32 scenes. These secondary results show dependence on dynamics and task assistance; IID pushing success does not establish general object manipulation.

### Retention and prediction diagnostics

The frozen mature Baby scores 80.90% on the separate IID retention panel. After adapting to `delay_3`, full-update success is 41.67%, replay 78.47% and low LR 80.90%; after `delay_6`, these are 23.26%, 59.03% and 74.31%. This descriptive pattern is consistent with losing previously effective control during adaptation, but is not an exclusive causal proof of forgetting.

The archived report also records reduced eight-step free-running hand-position RMSE after full updates: for example, 219.9 to 158.9 mm in `delay_3`, and 353.5 to 216.1 mm in `delay_6`, despite lower task success. Thus **lower prediction error in these diagnostics does not imply better control**. These are secondary diagnostics from the [supplemental report](../../results/raw/early/supplement_report.html); this documentation pass did not recompute them from every trajectory-level prediction record.

## 5. Supported conclusions and limits

Retain the following: the earlier body-prediction controller shows saved-result evidence of learning progress and bounded physical-condition transfer; naive mature-model adaptation can degrade performance; replay partly mitigates that degradation; and a finite-history engineering baseline is stronger under this protocol. Further work should explain how prediction quality, history, control rules and experience coverage jointly affect success.

Do not extend these findings to language contribution without language/audio ablations; scaling laws without controlled model/data/compute sweeps and fitting; consciousness or autonomous goal formation; real robots or biological infants. The finite-history result does not decide the general value of world models or multimodal learning. The later world-model experiment has different tasks, inputs and a distinct training defect; its numbers must not be concatenated with these curves.

## 6. Sources and verification scope

- Main study: [completed state](../../results/raw/early/formal_suite_state.json), [locked protocol](../../results/raw/early/formal_protocol.json), [statistical summary](../../results/raw/early/formal_summary.json), [methods](../../results/raw/early/formal_methods.md).
- Supplement: [locked protocol](../../results/raw/early/supplement_supplement_protocol.lock.json), [runtime](../../results/raw/early/supplement_runtime.json), [methods and limits](../../results/raw/early/supplement_methods.md).
- The original 2026-10-23 HTML is preserved as an [archived analysis](../../results/raw/early/archived_2026-10-23_analysis.html). Its tables are traceable, but the scope qualifications in this document take precedence. The archived report is not a fresh untouched test.
- Complete raw visual/audio trajectories and checkpoints remain in the original result repositories. This package carries compact evidence, not all sensory recordings. See [provenance.json](../../results/raw/early/provenance.json) for locations and source-file digests.
