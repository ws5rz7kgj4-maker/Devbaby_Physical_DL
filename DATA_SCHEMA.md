# Data contract and information boundary

The physical formal suite stores base runs under `members/<variant>-s<seed>/` and
adaptation recordings under `adaptation/s<seed>-age<block>-<domain>/experience.zip`.
The observation contract below is the same for both. Delayed worlds record issued
commands in the training array, and the applied delayed command separately in
annotations. Body mass, servo gain/damping, command displacement scale and delay
are annotated in `scene_and_language.json` under `effector`.

Physical evaluation `trials_*.jsonl` records the explicit task goal, commanded action
sequence, observed body trajectory, physical-contact flags, success at every step,
the final verdict and scoring-only object positions. The learned controller never
receives the scoring-only trajectory or simulator metadata. Development, final and
adaptation-test namespaces are distinct and saved in the protocol.

## Observable training arrays

Each `data/episodes/NNNNNNN/observations.npz` contains **exactly**:

| Key | Shape | Meaning |
|---|---|---|
| rgb | [T+1,64,64,3] uint8 | Infant-camera RGB at time t Δt |
| audio | [T+1,32,16] float32 | Fixed log-mel transform of audio heard in the preceding interval; row 0 is silence |
| proprio | [T+1,10] float32 | hand x/y/z, clipped vx/vy/vz, grip closed, contact, head yaw, holding-contact flag |
| actions | [T,5] float32 | x/y/z effector displacement increments, grip command, head-yaw increment, each in [-1,1] |

An action at t changes the world over [t Δt,(t+1) Δt). The next RGB, proprioception
and audio block become available at (t+1) Δt. The model predicts them using observations
available by t. It never receives audio from the next interval as an input.
The audio loss target may use that future interval, as ordinary next-step supervision.

The 16 kHz mono PCM `audio.wav` covers [0,T Δt). It includes all synthetic speech,
impact sounds and silence. The video displays frame t over that interval and uses this waveform.
Raw waveform and raw pixels are retained even though training uses mel features and small images.

The data loader enforces an exact allow-list; adding a `label`/`answer` array makes it fail.
No IDs, segmentations, bounding boxes, object positions, category strings, scripts,
utterance timestamps or word boundaries enter the network.

## Annotation-only files

`annotations/scene_and_language.json`:
- independent physics/language/sound random seeds derived from episode seed;
- layout theme, gravity, coordinate and time units;
- object IDs, color, shape, dimensions, material, mass, friction, initial position;
- utterance text, onset/offset, voice/rate/pitch, intended referent when grounded;
- spoken color/shape/event, current pointing target, grounded vs unrelated sampling flag.

`annotations/frames.jsonl`, one JSON per observation:
- timestamps; all object positions/quaternions and linear/angular velocities;
- rendered visibility pixel counts and 2D bounding boxes;
- moving/falling labels defined by explicit velocity thresholds;
- collision body pairs, contact force, within-step timing;
- infant grasp/release and caregiver external-force interventions;
- effector position, head angle, grasped body, actual action.

TTS onset/offset refers to the full generated utterance. If it extends beyond the end of
an episode, only its intersection with the recorded time range is heard. Text is never
substituted for the waveform. Word/phoneme-level forced alignments are not supplied in v0.1.

All simulator-defined variables used in this experiment are recorded, not every imaginable
semantic property of a physical world. A 3D cylinder has no hidden human annotation such as
"a cup". This baseline does not model liquids, soft bodies, hunger, pain or emotions.

## Randomization and splits

Colors: red, blue, green, yellow. Shapes: ball, block, cylinder.
Global training holdouts: red ball, blue block, green cylinder, including distractors.
Training samples the other 9 combinations without a developmental curriculum.
Physics, body exploration, caregiver phrasing and voice are stochastic.
Words retain their meaning within an experiment; unrelated q=0 utterances are independently sampled.

Training voices: `en-us`, `en`; held-out voice: `en-gb-scotland`.
Rate and pitch vary independently of referent. These are eSpeak NG formant voices.
Three training room themes and one held-out arrangement use procedural rigid geometries.
Held-out physics changes gravity, masses and friction ranges; it is not a test of reality transfer.

Test materials are generated from a separate seed namespace. The same fixed test items are
used across checkpoints and independent training seeds. They never enter replay or optimizer updates.
For IID-like object tests all candidate combinations are drawn from the training support.
Combination tests use a held-out target with color-only, shape-only and combined distractors.

## Replay and continuation

Each episode is recorded once. Each update's sampled episode and start index is logged.
Clips preserve causal time order. Replaying a clip is a training repeat, not a new environment interaction.
The growing weights carry experience across episodes; recurrent state resets at episode/clip boundaries.
Counters distinguish environment transitions from gradient updates.

Checkpointing is at complete block boundaries. Resume reuses previously recorded episodes,
restores weights/optimizer/RNGs and truncates stale partial-block update logs.
All raw experience shards are retained, including experiences no longer sampled by rolling replay.
Deterministic replay has been checked locally. GPU kernels, different hosts and restarted TTS engines
may prevent cross-machine bitwise equality; recorded waveforms are the authoritative observations.
