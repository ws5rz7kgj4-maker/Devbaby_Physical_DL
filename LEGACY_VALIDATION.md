# Validation record — 2026-10-23

## Executed here

- Python 3.11.17, PyTorch 2.6.0+cpu, PyBullet 3.2.7, eSpeak NG loader 0.2.4.
- Full test suite: **17 tests passed**. The report plotting dependency emitted
  non-fatal Pyparsing deprecation warnings.
- Additional annotation-schema regression check after adding static scene/body/camera metadata: passed.
- Full smoke pipeline: procedural 3D scene generation, audible formant speech, impact synthesis,
  raw episode recording, training from random weights, initial and subsequent frozen evaluation,
  optimizer/model checkpoints, report curves and an MP4 with an audio stream.
- Smoke budget: 4 episodes, 96 environment transitions, 8 optimizer updates.
- Learning-progress exploration branch was also run end to end at smoke size.
- An intentional interrupted-training integration test restored weights, optimizer and replay RNG.
  With the same recorded inputs, final parameters and update trace matched uninterrupted training exactly.
- Hugging Face launcher was exercised with dry runs and mocked API tests for duplicate prevention,
  lost server response recovery, secret handling, and source-file allow-listing.
- Multi-run analysis script was run on the smoke outputs.

## Tests explicitly cover

1. Speech waveforms have nonzero energy and distinct supported synthetic voices.
2. Actual rigid-body motion under actions; train/held-out attribute disjointness.
3. Seeded physics/render replay determinism in this CPU environment.
4. Synchronized recording shapes and annotation-only semantic labels.
5. Altering annotation files does not change training batches.
6. Neural gradients update weights; evaluation leaves parameters unchanged.
7. Removing actions makes the dynamics prediction invariant to changed action inputs.
8. Resetting world memory removes dependence on earlier world observations at that step.
9. Extra semantic fields in the observable NPZ are rejected.
10. Optimizer/checkpoint round-trip reproduces a subsequent training update.
11. Progress exploration accepts observable sensors only and has a finite training loss.
12. Familiar-condition candidate objects exclude held-out combinations.
13. Evaluation data are identical across different training q/delay settings.
14. Repeating a launch does not call the mocked Job-creation API twice.
15. A simulated lost submission response does not create another paid Job.
16. Dry-run and source allow-list checks prevent unintended submission/file inclusion.
17. Full interrupted-training continuation matches the uninterrupted reference.

Some test functions contain more than one assertion group; the actual pytest count is 17.

## What this does not establish

- No real Hugging Face Job was submitted in this chat; no user GPU credit was spent here.
- CUDA execution, HF account permissions, actual remote artifact upload and network behavior
  remain to be verified by the user's first smoke Job. The image and Jobs API were checked
  against official documentation; the pinned library's call signatures were inspected locally.
- No full pilot/research experiment matrix or multi-seed scientific result is being claimed.
- Four object-test trials per condition in smoke are deliberately tiny. Their percentages
  are not evidence that the model learned language, generalized, or developed consciousness.
- Exact continuation was tested for fixed recorded input on one CPU environment. Cross-host
  physics/audio/kernel bitwise equality is not promised.

## Included evidence

`examples/smoke/` contains the measured smoke report, metrics, raw recorded episodes,
audible video and model snapshots. These snapshots are examples only; new jobs initialize
fresh random weights and never load these examples automatically.
`verification/` contains the test output and a small provenance summary.
