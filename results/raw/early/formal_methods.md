# Locked physical-development protocol

## Research question and scope

Does self-supervised learning from entirely synthetic, random sensorimotor experience
improve executed physical-task success in unseen physical distributions? Does later
development change the gain obtained from a fixed amount of new-distribution experience?

This is an operational test of a finite model, body, planner and task family. It does
not establish general intelligence, subjective experience, language understanding,
spontaneous goal formation, or autonomous acquisition of the supplied task cost.

## Learner and information boundary

Approximately 1M random parameters; raw 64x64 RGB, acoustic features derived from PCM,
proprioception and issued actions. Self-supervised prediction/reconstruction and
audiovisual contrastive objectives. No pretrained weights, transcripts, object labels,
success rewards, or object poses enter model fitting. Random movement uses no object labels.

The evaluator specifies a static spatial/contact goal. For touch/lift this includes
the initial target contact point. A supplied CEM planner scores action sequences
using the learned prediction of bodily consequences and the explicit task objective.
The body includes a proximity grasp primitive. Neither the planner nor the task cost
is learned; physical movement prediction is. Learned planning cannot read live object
poses, mass, friction, obstacle geometry or success. These are recorded for analysis.

## Base development

Two conditions, full and action-input ablation; seeds 0..7. Same architecture size,
optimizer settings, data budgets, scene seeds, motor commands and initial weight seed.
The no-action model still sees motion through proprioception; it cannot condition its
transition predictions on chosen actions. This is an intentionally direct mechanism
ablation, not a claim of superiority over modern robotics methods.

512 episodes × 64 interactions = 32,768 interactions per model. 16 collection blocks,
512 updates each = 8,192 updates. Training episodes are independent random scenes;
the model weights persist. No curriculum or performance-based early stopping.
Episode resets limit claims about lifelong continuous memory.

Base model total: 524,288 transition exposures and 131,072 optimizer updates across
16 models. Within a seed, the two random-experience conditions reuse immutable
recordings: 262,144 unique generated base transitions when the cache is available.
The replay loader only exposes episodes up to the current developmental block,
even when later episodes are already cached. Physical regeneration after loss of
cloud scratch can add engineering cost without changing the exposure budget.
Full-model adaptation: 8 seeds × 3 developmental checkpoints × 2 domains × 2,048
interactions = 98,304 additional interactions. Adaptation experiences do not update
the base checkpoints. Evaluation interactions are additional, frozen, and excluded
from both training-exposure axes; they still consume compute.

## Development, final test, and adaptation separation

Development panel: fixed at 0/8192/16384/24576/32768 interactions. Six trials per
task/domain/seed; common scenes enable paired growth measurements. No checkpoint is
selected by scores. All per-trial outcomes, including failures, are retained.

Sealed final panel: fresh deterministic namespace, 24 trials per task/domain. First
evaluated only after all base models reach the fixed endpoint. Both the initial and
endpoint full-model snapshots use this panel for the paired final comparison.

Adaptation: checkpoints at 0/16384/32768 interactions, full model only. Each clone
has a fresh optimizer and replay. Random experiences in new_dynamics/new_delay at
0/128/512/2048 added interactions, 16 updates per 64-step episode. Data seeds and
batch-sampling schedules match across ages. Separate namespace for evaluation.
No success reward or evaluator trajectory enters adaptation. An interrupted clone
restores its own optimizer/RNG and never substitutes later weights for earlier budgets.

## Success and measurements

Reach: within 8 cm for the final 3 steps. Touch: actual contact for any 3 consecutive
steps. Grasp-lift: target held, object center at least 14 cm above its initial height,
hand within 13 cm of the lift goal, all for the last 3 steps. Fixed horizon: 40 steps.
All trial criteria are defined before training and are not calibrated on outcomes.

The OOD macro score equally weights tasks and the four unseen domains: changed
dynamics, changed actuator delay, changed obstacle layout, and all three together.
IID performance is separate. Task-wise/condition-wise scores must accompany the macro
score, so easier reaching cannot silently hide failed grasp-lift performance.

Growth rate: 100,000 × [P(n_i)-P(n_{i-1})]/[n_i-n_{i-1}], in percentage points per
1,000 environment interactions. This is a finite-difference rate, not an inferred
derivative or proof of a phase transition. Negative slopes remain negative.

Learning efficiency: normalized trapezoidal AUC, plus the first of two consecutive
scheduled checkpoints at/above 50% and 80%. Non-attainment is null/censored, not zero.
Adaptation also reports AUC of P(b)-P(0), removing the initial success offset. This
baseline subtraction does not remove ceiling effects; raw curves and starting
success levels remain essential. A higher unadjusted AUC alone is not evidence of
faster learning.

## Primary statistical family

Three predeclared paired-seed contrasts:
1. Full-model final OOD success minus its random-initialization OOD success.
2. Full-model final OOD success minus no-action endpoint OOD success.
3. Final-age adaptation gain AUC minus initial-age adaptation gain AUC, averaged over
   the two adaptation distributions.

Two-sided exact paired sign-flip tests, Holm correction across these three comparisons,
and paired-seed percentile bootstrap 95% intervals (10,000 resamples). Tests assume
exchangeability/symmetry of seed-level differences under the null. Training seeds,
not video frames or repeated checkpoints, are the independent resampling units.
Intervals are conditional on the fixed sampled test panel; they do not include an
additional resampling hierarchy over scene instances or physics distributions.
Eight seeds are a budget choice, not a completed prospective power calculation.
Effect sizes and uncertainty are required even if adjusted p-values are small.

Development slopes, thresholds, task/domain breakdowns and intermediate-age adaptation
are descriptive secondary analyses. No adaptive pruning or test-score-based tuning.
Changing the protocol after looking at these final tests starts a new exploratory
experiment and requires new final-test seeds for another confirmatory claim.

## Compute and persistence

One HF Job installs dependencies once and runs members sequentially. Operational
wall-clock interruption is independent of accuracy. Only complete members count;
partial suites cannot produce a completed formal summary. Resume skips completed
models and completed tests. Immutable raw recordings are uploaded before local
scratch deletion. A hard infrastructure loss may lose the unfinished block or the
unuploaded portion of an adaptation budget; no cross-host bitwise guarantee is made.

## Feasibility controls and limits

Random controller and privileged scripted reference are simulated on the same final
panel. The reference is not an optimality bound and can fail, especially under delay
or obstacle changes. Its failures are reported, not filtered out after observing model
performance. Material reference failure indicates benchmark/controller limitations.

The tasks use a simplified point hand, one explicit target and short episodes. Positive
results do not establish general manipulation, unseen task families, language-driven
control, realistic infant development or real-world transfer. Negative results may
reflect perception, dynamics, planner horizon, task objective or exploration coverage;
the retained data/annotations support diagnosis without altering the frozen scores.
