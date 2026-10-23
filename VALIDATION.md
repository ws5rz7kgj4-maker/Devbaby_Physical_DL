# Physical experiment code validation — 2026-10-23

Environment: Python 3.11.17, PyTorch 2.6.0+cpu, PyBullet 3.2.7, eSpeak NG loader 0.2.4.

Executed software checks cover:

- Audible non-neural speech synthesis and physical rigid-body motion.
- Full recording schema and semantic-annotation isolation from training.
- Matching physical/visual experiences when changing speech correspondence.
- Out-of-training actuator mass/gain ranges and actual delayed command execution.
- Action ablation, frozen evaluation, and controller independence from annotation fields.
- Executed reaching/contact/grasp-lift evaluation with fixed trial outcomes and traces.
- Exact CPU training continuation for fixed recorded inputs.
- Adaptation clones, fixed-budget evaluation, completed-adaptation reuse, and interrupted
  adaptation recovery without overwriting earlier-budget scores with later weights.
- Growth-rate units, negative growth, censored thresholds, seed-level tests and Holm correction.
- Higher initial success does not count as an adaptation learning gain.
- A partial experiment matrix cannot be labelled a completed formal result.
- Complete physical suite, including two conditions, two seeds, development snapshots,
  separate final tests, adaptation, statistical CSV/JSON, plot and HTML report, at tiny
  software-test budgets. Re-running with resume reuses completed work.
- Launcher dry run and mocked submission idempotency, source allow-list, and lost-response handling.

The final combined regression run passed **28 tests** in 71.03 seconds. The logs
shipped with this release are authoritative. Additional engineering feasibility
checks execute the privileged reference at the real 40-step horizon on a separate
namespace, including delayed actuators and load compensation; they are not learned
agent results and are not the formal final-test panel.
`examples/physical_functional` contains actual tiny-budget software-test output. It is
explicitly marked **FUNCTIONAL TEST ONLY — NOT RESEARCH RESULTS**. Its percentages
must not be used to claim physical generalization or learning improvement.

No user HF Job was submitted here, and no paid cloud compute was used. The actual
24-hour GPU Job, account authorization, upload throughput and CUDA behavior are not
validated by these CPU checks. The HF API and container choice were checked against
official documentation; the launcher uses the pinned huggingface_hub 0.36 API.

The final 16-model experiment and its adaptation matrix have not been run here.
The package contains runnable experiments, not invented final performance curves.
Bitwise equivalence across different physical simulators, machines or GPU kernels
is not promised.
