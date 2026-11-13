# Scope of the physical formal experiment

The primary outcome is **executed physical success on unseen physical conditions**.
The second objective is how that success and the additional benefit of adaptation
change with accumulated training experience. See `FORMAL_EXPERIMENT.md` for the
locked budgets, splits, criteria, metrics, tests and limitations.

Implemented tasks: reach a supplied coordinate, physically touch a target, grasp and
lift a target. The controlled body is a 3D point hand with a proximity grasp primitive
and head yaw. Spatial/contact goal specifications, CEM search, and its task objective
are supplied. A randomly initialized neural sensory world model learns the movement
prediction used by this planner. The system does not autonomously invent its tasks.

All training data are simulated, including visible caregiver cues, audible formant
speech and physical collision sounds. The main data stream has only RGB, acoustic
features, proprioception and commands. Hidden simulator properties and language
annotations are stored separately. No pretrained model, ASR, language tokenization,
object coordinates, target success rewards or transcripts enter model fitting.

The neural loss is next-frame MSE + current-frame reconstruction MSE + 0.2 latent
MSE + 2 proprioceptive MSE + 0.1 acoustic-feature MSE + 0.25 audiovisual InfoNCE.
The target visual latent is stop-gradient; reconstruction anchors visual features.
This is a predictive baseline, not a reproduction of a named JEPA method.

The formal suite does not treat lower prediction loss, decoded attributes or speech
matching as physical success. It runs the learned planner in actual simulator
episodes and records all issued actions, sensory/body trajectories and outcomes.
Repeated development panels characterize growth; a separate final panel is used
after all fixed-budget training finishes. Adaptation uses separate model copies.

The formal controls include an independently trained action-input ablation, the
initial full-model snapshot, a random actor and a clearly labelled privileged
scripted feasibility reference. The latter is never reported as learned behavior.

Success curves, finite-difference growth rates, learning AUC, censored threshold
crossings, and adaptation gain AUC measure different things. In particular, higher
zero-shot starting performance is not by itself faster learning. Baseline-adjusted
adaptation gains are reported alongside raw curves and possible ceiling effects.

The scope is short, explicit-goal control with a simplified body and a finite set
of held-out dynamics/layout changes. It does not establish open-ended manipulation,
new task-family discovery, language-generated plans, spontaneous speech, human-like
development, consciousness, or transfer outside the simulator.

Increasing budget may or may not improve performance. Failed tasks and negative
growth must be reported. The formal protocol is a reproducible experiment, not a
promise of a positive finding or publication-level novelty.
