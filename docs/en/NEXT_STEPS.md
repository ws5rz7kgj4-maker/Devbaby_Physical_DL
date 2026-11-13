# Limitations, research value and next steps

[中文](../zh/NEXT_STEPS.md) · [Home](../../README.md)

## Value of the completed work

The evidence supports three concrete outputs: a traceable fully synthetic multimodal pipeline; early restricted-control results showing improvement through training but possible deterioration through continued adaptation; and reproduction/localization of a later world-model failure in which parameters never trained. These support an open research record and a reproducible technical report.

The useful next question is: **Which interaction experiences, state histories and update procedures turn improved prediction into physical success in new situations?** Earlier supplementary diagnostics already show that lower prediction error can accompany worse control. Reducing reconstruction loss or adding updates alone does not guarantee better task performance.

No new literature-novelty review was conducted for this snapshot. It does not establish first discovery, a solved field-wide bottleneck or publication acceptance. History/replay advantages remain protocol-specific, and simple feedback control is strong.

## Open questions

| Question | Existing evidence | Missing evidence |
|---|---|---|
| Does physical success grow with experience? | Complete early body-control curves | Repaired image-world-model curves |
| Do mature models learn faster? | Early naive adaptation deteriorates overall | Reproducible positive transfer and fair speed comparisons |
| How does random exploration affect useful experience? | Pilot contact/displacement coverage | Formal training, interventions, information metrics and downstream success |
| Why does history help? | Stronger supplementary history baseline | Parameter/target/compute-matched controls and input/controller disambiguation |
| Does language help physical generalization? | Synthetic speech input exists | Valid speech/impact/silence/shuffle comparisons; early studies lack a language-specific ablation |
| Is this an effective world model? | Action-conditioned prediction/planning prototype | Reliable repaired prediction and closed-loop success |
| Is there a scaling law? | Several interaction-budget checkpoints | Multi-scale design, prespecified fitting and held-out-scale validation |

“Useful operational information” does not yet have a validated information-theoretic definition here. Contact, displacement and event coverage are proxies and must not be relabeled entropy, mutual information or controllable information.

## Gated next experiments

1. **Full training-path regression.** Use the real model, batch, AMP settings, original `optimizer_update` and GradScaler. Check presence/finite gradients for every expected parameter and cumulative updates over several normal steps. Inject overflow to verify same-batch/RNG retries and FP32 fallback. Test model, optimizer, scaler and RNG preservation across save/resume. A zero gradient in one step need not be a defect; unexplained persistent non-updates require investigation.
2. **Small-data learnability.** Start from random initialization and establish tiny overfit and action-conditioned prediction. Separate object, motion, hand/body and background metrics; include persistence and independently simulated action counterfactuals rather than treating input mismatch as causal prediction. Track every module's changes.
3. **Simple closed-loop tasks.** On fixed development scenes, verify that predictions help simple relocation/lifting. Report random, simple feedback, hindsight BC and privileged references. Preserve sensory/action/prediction/outcome traces; establish task feasibility and correct control interfaces before increasing scale.
4. **Lock a new protocol.** Use new training/validation/test namespaces and prespecify seeds, scenes, stopping rules, effect thresholds and compute budgets. Previously inspected test data become development data. Discard compromised learned weights; synthetic trajectories may be reused after split checks.
5. **Study exploration mechanisms.** Manipulate waypoint persistence and noise while measuring realized action entropy/correlation/saturation, contact and repeatable physical effects. Match interaction budgets and distinguish coverage from update count. Compare held-out success curves, areas, interactions to a prespecified threshold, and retention during adaptation.
6. **Consider scaling only after competence.** Once stable performance escapes floor/ceiling effects, vary data, model size and compute independently and test predictions on held-out scales. No power-law exponent is fabricated for the current evidence.

These are pending experiments. Preparing this documentation does not complete them or submit cloud jobs.

## Publication scope

The current material suits an engineering/research progress release. A paper should distinguish completed early control/adaptation studies from the later defect audit, documenting task assistance, baselines, sampling units, reused scenes and code versions. Claims that random exploration improves general physical world models require effective repaired training, multi-seed confirmation, independent new situations, strong baselines and mechanism ablations. Complete code alone cannot replace these results.
