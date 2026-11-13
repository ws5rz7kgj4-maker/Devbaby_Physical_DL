# Methods and metric definitions

[中文](../zh/METHODS.md) · [Home](../../README.md)

The project contains distinct model generations. See [early experiments](EARLY_EXPERIMENTS.md) for body prediction and reaching. The architecture, tasks and diagnostics below primarily describe `DevBaby-worldmodel-v2`. Results from these generations must not be joined into one learning curve.

## Research question

How do exploration persistence and noise change contact and physical-state coverage? Do those experiences help a randomly initialized learner complete tasks in new physical situations and improve learning speed? Speech and sound are co-occurring inputs whose contributions require controlled comparisons.

Developmental age means the cumulative interaction prefix available to training, not biological age, language proficiency or consciousness. The intended primary outcome is physical task success on independent situations. Prediction error, contact coverage and candidate ranking are distinct auxiliary outcomes.

## Fully synthetic, with explicit design assumptions

PyBullet generates rigid balls, boxes and cylinders with randomized mass, friction, restitution and layout. The body is a force-controlled spherical end effector with a programmed proximity-grasp constraint, not articulated fingers or a biological infant.

Two fixed cameras generate RGB. Impacts are analytic damped oscillations; caregiver speech uses eSpeak NG. The current version randomizes the object, English template, voice, speed, pitch and interval while grounding color, shape and material words in scene properties. This is **randomized, semantically grounded synthetic speech**, not words sampled independently of the world. Templates, sensors, embodiment and tasks are designed priors; random initialization does not make the system assumption-free.

The learner has no pretrained language, vision or audio weights and receives no text, tokenizer output or transcripts. Learning acoustic associations would not establish language understanding, spontaneous speech or consciousness.

## Data and annotation boundary

A 128-step episode has 129 observation times. Control runs at 8 Hz, physics at 120 Hz and raw audio at 16 kHz.

| Content | Shape / meaning | Used by world-model optimization |
|---|---|---|
| RGB | `[T+1, 2, 64, 64, 3]`, uint8 | Yes |
| Proprioception | `[T+1, 10]`: position, velocity, closed, contact, yaw, held | Yes |
| Action | `[T, 5]`: translation, grip, yaw | Yes |
| Audio features | Mixture/impact/speech, each `[T+1, 32, 16]` log-mel | Selected by arm |
| WAV | Three channels: mixture/impact/speech | Archived; learner uses features |
| Object truth | Pose, velocity, presence, contact, segmentation | No |
| Scene/language labels | Materials, shapes, dimensions, seeds, cameras, utterances | No |

`actions[t]` is issued after observation t and may affect later observations through a delayed actuator. Audio at t+1 summarizes the preceding interval. Transcripts and truth are stored separately under `annotations/`.

Frozen-representation ridge probes do fit to labels. Task construction, scoring and privileged references also use simulator truth. The correct boundary is that world-model optimization and learned planning do not consume object labels, rather than that labels are unused anywhere in the experiment.

## World model and planning

A CNN encodes two images concatenated into six channels, another encodes audio, and the features are fused with proprioception. Hidden size is 192 and audio embedding size is 48. A `dynamics` GRUCell advances the latent state from a five-dimensional action; a `filter` GRUCell incorporates observations. Decoders predict images and ten proprioceptive channels.

Losses combine posterior reconstruction, one-step and multi-step prediction up to 32 steps, image-motion weighting and a small latent-spread penalty. There is no task reward or object-label loss. The original protocol uses batch size 8, burn-in 8, 32 episodes and 256 updates per block, for 16 blocks: 65,536 interactions and 4,096 updates per seed. Data may be generated in advance, but replay access is restricted to the current age prefix.

The planner receives current observations and final two-view goal images. CEM searches action chunks repeated for four steps. Planning horizons are 8/32/64, with a nominal budget of 2,048 latent transitions per iteration and three iterations. Longer horizons have fewer candidates and more action variables; this is not an isolated horizon effect or exact FLOP matching. The 64-step horizon extrapolates beyond 32-step training.

## Original formal design is not a completed result

There are eight training seeds and five audio arms: `full / silent / impact_only / speech_only / shuffled`. Evaluation domains are `iid / new_physics / new_layout / occluded / material_shift`; tasks are relocation, lifting, two-object stacking, and arrangements of two or three objects. Audio, planning horizon and hindsight goal-conditioned BC comparisons are fixed in the protocol.

BC constructs goals from future images in random trajectories and regresses recorded actions; it has no expert demonstrations. The privileged scripted controller can inspect truth, actuator queues and goals. It is an engineering feasibility reference, not a learned agent.

Physical success requires all designated objects within 0.075 m of their goals for the final three consecutive times. Lifting also requires holding; other tasks require release; stacking requires contact between designated objects. A privileged reference or privileged candidate trajectory must not be counted as learned success.

The original plan specifies paired seed/scene comparisons, a crossed bootstrap and Holm correction for six primary contrasts, withholding complete inference when required cells are missing. The run was subsequently canceled and a training defect was discovered. These are **design specifications**, not completed valid inferential results.

## Distinct delay conditions

The current `new_physics` domain changes several dynamics properties, including FIFO motor-command delay. IID delay is zero or one control step (0/125 ms); new physics uses three steps (375 ms). It does not delay rewards or labels and does not isolate delay as the only intervention.

Legacy code also offers caregiver-speech delay, but it is not an executed condition in the archived early formal/supplementary studies; the early formal protocol sets it to zero. See [early experiments](EARLY_EXPERIMENTS.md) for executed motor-delay conditions. The current `PhysicalWorld._speak()` overrides the legacy speech-delay mechanism. This code option must not be presented as a completed speech ablation or one of the current five audio arms.

## Diagnostic definitions

| Metric | Definition | Does not substitute for |
|---|---|---|
| Contact time fraction | Steps with object contact / episode steps | Task success or information |
| Action-effect time fraction | Fraction with any object displaced >2 cm relative to the matched idle trajectory | New information per step, entropy or mutual information |
| Object-region MSE | Normalized RGB MSE on the union of visible object masks at the initial and target times | Position error or control success |
| Persistence | Predict the initial image, with the same target and mask | Learned dynamics |
| Candidate distance advantage | Mean true candidate endpoint distance minus selected candidate distance, in metres | Full CEM success rate |

Action effects include persistent displacement after an interaction. The pilot uses a 0.04 m/s movement threshold, whereas original coverage statistics use 0.03 m/s; these movement rates must not be pooled directly.

Action-mismatch diagnostics compare actual, zero and reversed actions. Reversal applies to the full future 64-step sequence before taking a horizon prefix; it is neither sign negation nor reversal of each horizon's original prefix. All variants are scored against the actual-action target image. Counterfactual outcomes were not simulated for these mismatched sequences.

## Generalization, learning speed and scaling laws

A future confirmatory study should fix new training/test seeds and evaluate multiple interaction budgets on paired scenes. Report success, area under the success curve, interactions to a prespecified threshold, and adjacent-stage change per interaction separately, retaining negative growth and unreached thresholds. Set thresholds and minimum meaningful effects before final testing.

A few age checkpoints do not constitute a scaling law. No validated relationship across model size, data and compute has been fitted, and repaired world-model physical-success curves remain unavailable.
