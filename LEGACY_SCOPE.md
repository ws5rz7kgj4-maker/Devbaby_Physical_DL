# What the experiment tests

## Implemented model and training

All learned modules start at random weights: visual CNN, log-mel audio CNN, audio and
visual temporal encoders, multimodal recurrent world state, action-conditioned prediction,
pixel reconstruction, proprioceptive prediction and cross-modal contrastive projections.
There is no LLM, ASR, pretrained visual encoder or semantic tokenizer in the infant.

Loss = next-frame pixel MSE + current-frame reconstruction MSE + 0.2 latent MSE
+ 2 proprioceptive MSE + 0.1 audio-feature prediction MSE + 0.25 audiovisual InfoNCE
+ optional 0.2 learning-progress prediction MSE.
The target latent is stop-gradient; pixel reconstruction anchors the visual encoder.
This is a simple predictive baseline, **not an implementation/reproduction of a named JEPA paper**.
No claim of biological realism or architectural novelty is made.

Audiovisual positives are same-time fixed-duration clips, selected without transcript
boundaries or referent labels. The learner must handle ambiguous multi-object scenes.
The acoustic-energy gate is computed from observed waveform features, not speech annotations.
InfoNCE can contain false negatives (two clips about the same concept); this is a limitation,
not something silently corrected using ground-truth labels.

Random exploration samples spatial motor waypoints independently of the objects and labels.
The progress variant trains an interest predictor on the difference between an older model's
and the current model's proprioceptive prediction error, measured on observed transitions.
It chooses among random candidate actions with 35% random exploration. This algorithm is
experimental, not guaranteed to outperform random exploration, and is not a task-solving policy.

There are no manually scheduled infant ages, semantic stages or curriculum phases.
The same objectives operate throughout. Collection and optimization alternate in blocks
for practical batching; this is not a claim of exact real-time biological online learning.

## What each score means

| Score | Supported interpretation | Unsupported interpretation |
|---|---|---|
| Four-way audio/object retrieval | A learned cross-modal compatibility function distinguishes candidate rendered objects | Full natural language understanding or instruction following |
| New color×shape targets | Limited compositional transfer within a small artificial vocabulary | Open-vocabulary generalization |
| New voice/layout/physics | Robustness to specified synthetic changes | Generalization to real children/robots/worlds |
| Event diagnostic | Matching speech to controlled still/moving/falling clips | Pure causal reasoning; static height cues can help |
| Next-frame/body prediction | Improved sensory consequence prediction relative to persistence/shuffled-action baselines | Intent, awareness or consciousness |
| Executed reaching | Learned dynamics helps a planner move a simplified hand toward an explicit coordinate goal | Verbal commands autonomously grounded into manipulation goals |
| Frozen-feature ridge probe | A researcher can decode an annotated property from the representation | The infant can itself report or use that property |

The motor analytic reference knows the control mapping, not object labels. It is a diagnostic
upper reference for this simplified body, not a learned peer baseline. The random comparator
has the same environment seeds and body goals. Do not report oracle success as model success.

`no_world_memory` removes only recurrence in the world-state module; acoustic and visual
clip encoders retain short temporal context. It is not a total-memory ablation.
The `q0` control samples noun attributes independently with the same combination prior.
Verb frequencies may differ because related utterances follow actual motion states; the
matched cross-trajectory `shuffle_audio` control is therefore also included.

## Recommended analysis plan before training

1. Validate smoke, then run pilot and inspect audio/video and all failures.
2. Choose budgets, conditions, seeds, metrics and any onset thresholds before full runs.
3. Use at least five independent training seeds for comparisons where feasible.
4. Plot every checkpoint by actual environment exposure; do not select only the best checkpoint.
5. Treat same test scenes/frames within one seed as repeated measurements, not independent models.
6. Report zero-shot frozen evaluation separately from supervised diagnostic probes.
7. For causal claims about exposure, manipulate q/delay/exploration while matching budgets.
   Observational correlations between a particular event count and learning are not causal proof.
8. Keep final test conditions separate from development decisions. This code's repeatedly
   viewed benchmark is a development benchmark; create a fresh held-out seed set for a final paper.

Within-run Wilson intervals are descriptive uncertainty over test items. Cross-seed bootstrap
intervals use model seeds as sampling units; small seed counts remain uncertain. Multiple
conditions and repeated checkpoint inspection require a prespecified analysis or correction
before inferential claims. A threshold crossing does not prove a discontinuous emergence event.

## Important limits

- Simplified 3-DoF point hand, binary grip reflex and head yaw; no full infant skeleton.
- 64×64 vision, approximately 1M parameters, 2-second default training clips.
- Small English template language with colors/shapes and three motion descriptions.
- Fixed network capacity and learning rules; no neuron growth or autonomously invented grammar.
- No text/voice generation by the infant; caregiver audio is audible and recorded.
- No autobiographical self-report, consciousness classifier or claim of subjective experience.
- All data are synthetic; success says nothing by itself about transfer to real environments.
- Performance is an empirical outcome. More compute or more synthetic data does not ensure success.

Related scientific context (not training data):
- Smith & Yu, cross-situational word learning: https://doi.org/10.1016/j.cognition.2007.06.010
- gSCAN systematic generalization: https://arxiv.org/abs/2003.05161
- Theory-derived consciousness indicators: https://arxiv.org/abs/2308.08708
