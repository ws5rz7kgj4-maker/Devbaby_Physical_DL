# Supplemental protocol and interpretation

## Provenance and the new test

Original run: `formal-physical-eef2ff40`. Its locked protocol SHA-256 is `6d7277597bf82f318cb8db9b111cdc2c5af3962135f3ea7c940dbbfdc66b5976`; original source archive SHA-256 is `eef2ff40383ff64a46d1a0bc186a3829fc6502ce09700d5e7120e53ae65d8840`.

The launcher locates the user's original private dataset, pins its commit revision, checks the original protocol, finds the 24 full-model age snapshots and 128 raw-experience block shards, and verifies checksums. Eight trained Baby seeds and the age-zero random initializations are reused. Raw generated audio/RGB/action data are reused to train a new finite-history body-dynamics comparator. There is no language model, tokenizer, pretrained visual encoder, neural TTS model or external training corpus.

The original experiment's final results have already been examined. They are not presented as a fresh confirmatory test. Supplemental train, diagnostic, validation and final namespaces are separate. All supplemental model training finishes before final physical evaluation. Checkpoints are fixed by interaction budget; no selection by final success, no early stopping by outcome and no automatic hyperparameter search occur. If code/protocol/source changes, the existing run refuses resume; a different experiment ID and genuinely fresh test namespace are required after viewing results.

## Experience and annotation boundaries

Training episodes use random exploration independent of model and age. Adaptation datasets are shared across arms for a given model seed and domain. Replay uses only `rgb`, `audio`, `proprio`, `actions` from `observations.npz`. Scene properties, object poses, words, utterance alignments, issued/applied actuator actions and physical events are stored separately under annotations. Their presence enables analysis; they never enter the training loader or nonprivileged policy input.

Speech varies in utterance form, voice, timing and sampled scene referents, but remains grounded in the synthetic scene. Completely independent random words would remove the association being learned. The protocol randomizes from explicitly specified distributions, not from an undefined uniform distribution over all possible worlds or languages.

Original Baby receives RGB, mel-like audio features, proprioception and issued actions. It predicts sensory/body changes. Its forward pass resets GRU state at the start of each sampled clip and inserts zero for the first previous action; online control carries recurrent history. Diagnostics compare full-history context, reset with the true preceding action and reset with a zero preceding action. This difference is a testable training/control mismatch, not assumed to be the cause of every failure.

## Models and adaptation

Original snapshots correspond to 0, 16,384 and 32,768 interactions, and 0, 4,096 and 8,192 training updates. The finite-history comparator is a randomly initialized multilayer body-dynamics model. Each prediction uses eight `(observed proprioception, issued action)` pairs, including the current command. It does not receive the realized actuator delay or object truth. For 16-step clips, seven steps provide context and nine transitions are prediction targets. Its body-only objective, sensor restriction, parameter count and smaller number of valid targets differ from Baby. Comparisons control experiences and optimizer steps, not identical sampled minibatches, architecture, FLOPs or supervised-target counts.

Mature adaptation arms are:

1. `frozen`: no parameter or optimizer update.
2. `full_update`: original Baby loss with its original AdamW learning rate.
3. `low_lr`: same loss and data, learning rate multiplied by 0.1.
4. `frozen_encoder`: only world memory, visual dynamics and body prediction head update; perception, grounding and reconstruction modules are fixed.
5. `replay_mix`: half of each fixed-size minibatch comes from new experience and half from old training experience, with independent recorded sampling RNG states.
6. `history`: the finite-history comparator adapts its body predictor on new experience.

Budgets are 0/128/512/2,048 new interactions. Each 64-step episode permits 16 optimizer updates, matching the original supplemental ratio. All ages receive `full_update`; the five additional mature controls avoid a much larger full factorial. Older-experience replay never reads future development blocks. The total 2,048-step adaptation dataset is generated in advance, but the replay loader reveals only the episode prefix available at the current budget.

Three source ages identify interval-average performance growth, not an exact emergence time. Age-related learning gain/AUC is exploratory and does not by itself establish statistically confirmed learning-to-learn.

The primary six contrasts are mature final-budget full-vs-frozen, replay-vs-full, frozen-encoder-vs-full, low-LR-vs-full, history-vs-full and history-vs-frozen. Performance is averaged equally across reach, touch, grasp-lift and the three adaptation domains (push is separate exploratory task generalization). Difference signs are explicit in the CSV. Paired seed sign-flip tests use all eight independent trained seeds; Holm controls the six-test family. Crossed scene/seed bootstrap confidence intervals preserve matched conditions and task/domain weighting. Partial or unpaired results cannot produce final inferential conclusions.

## Physical scores and limits

Reach requires final-three-step hand distance under 8 cm. Touch requires three consecutive contact frames at any time. Grasp-lift requires target held at least 14 cm above its initial center and hand within 13 cm of the supplied lift goal in the final three frames. Push requires the object's horizontal position within 6.5 cm of the target, remaining on the support surface and without a grasp constraint in the final three frames. The full definitions are executable in `supplement_world.py`.

Both learned controllers use the same CEM planning procedure and public spatial goals. The new push task supplies a contact-then-hand-transport decomposition; neither model explicitly predicts object displacement. A successful push demonstrates performance under this assistance, not autonomous task discovery or general object-state planning. The body includes a simple programmed grasp primitive. Nonlearned feedback and random controls use available observations; the simulator-state oracle is labeled privileged and only assesses reference feasibility.

Different fixed delays share initial geometry. Delay zero still has inertial motor dynamics and is not instantaneous position teleportation. At 8 Hz, an action delay of six steps means the issued command begins affecting the actuator 0.75 seconds later. The initialized command queue contains zeros. Scored annotations retain both issued and applied commands, allowing the exact FIFO relation to be checked.

Zero-shot delay curves, new-geometry push tests, retention after adaptation, teacher-forced and CEM-matched free-running prediction errors and clip-context diagnostics are secondary analyses. Reduced prediction error is not automatically improved task success; the report keeps these distinct. Correlations or successful repairs suggest mechanisms but do not prove an exclusive causal explanation.

## Operational reproducibility

Source model files are trusted user-owned Torch checkpoints and require pickle-compatible loading; untrusted third-party checkpoints are not accepted as interchangeable inputs. Source commit, code digest and protocol are locked. Checkpoints save model/optimizer state and Python, NumPy, Torch, CUDA and replay RNG states. Physical evaluation saves completed trial records and uses deterministic scene/controller seed derivation so interrupted trials can rerun. A hard process kill can lose work since the last uploaded checkpoint, but does not justify replacing missing results with later weights. Cloud publication commits completed physical panels; a partly evaluated panel can be rerun on a new worker. Independent eSpeak synthesis may vary slightly across processes/cache state, so bitwise continuation refers to committed recorded sensory experience, not regeneration of an uncommitted speech episode. All arms share the same saved experience.

The original runtime dependency versions and PyTorch 2.6 CUDA image are retained; the terminal launcher has a separate HF Hub environment for current Jobs labels. Cross-device bitwise reproducibility is not claimed. All raw new experience is saved; evaluation sensory recordings can be shared across truly identical policy/panel evaluations. All source restores use pinned revisions. Final reports mark incomplete runs and count completed observations.

This code addresses a specified supplemental evidence package. It does not include a second simulator, a complete literature benchmark comparison, biological infant validation, a prospective power analysis or a test of consciousness. An empirical paper must limit claims accordingly and may need additional targeted experiments after results are known.
