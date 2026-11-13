"""Supplemental controllers using public observations and their own commands only.

HistoryDynamics is an explicitly specified proprioceptive system-identification
baseline, not a reproduction of a named published algorithm. Its training input
is the same recorded sensor/action data as Baby, restricted to proprioception
and issued actions. Simulator labels and the realized actuator delay are never
inputs. A window row is (p_t, issued_action_t), predicting p_(t+1).

The push objective supplies a contact-then-transport hand path. Neither this
baseline nor the legacy Baby controller explicitly predicts object displacement.
Object motion is scored separately by the environment; a successful push does
not demonstrate learned object-state planning or autonomous task decomposition.
"""
from collections import deque
import copy
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .physical import LearnedController
from .world import ACTION_DIM, PROPRIO_DIM


class HistoryDynamics(nn.Module):
    """Residual body predictor with an explicit finite command history.

    Eight entries include the present command and seven past issued commands,
    covering the supplemental benchmark's maximum fixed delay of six steps.
    All weights use ordinary random initialization. This is a smaller model
    than Baby; parameter count and update/interaction budgets must be reported.
    """
    def __init__(self, cfg, history=8, width=256):
        super().__init__()
        if history < 1 or width < 1:
            raise ValueError("history and width must be positive")
        self.cfg = cfg
        self.history = int(history)
        self.width = int(width)
        self.net = nn.Sequential(
            nn.Linear(self.history * (PROPRIO_DIM + ACTION_DIM), self.width),
            nn.LayerNorm(self.width), nn.SiLU(),
            nn.Linear(self.width, self.width), nn.SiLU(),
            nn.Linear(self.width, PROPRIO_DIM),
        )

    def predict_window(self, pairs):
        """pairs: (..., history, proprio+action), ending in the present pair."""
        if pairs.shape[-2:] != (self.history, PROPRIO_DIM + ACTION_DIM):
            raise ValueError("Incorrect history window shape")
        current = pairs[..., -1, :PROPRIO_DIM]
        delta = self.net(pairs.flatten(-2))
        return current + delta

    def forward(self, batch):
        pr, actions = batch["proprio"], batch["actions"]
        if pr.shape[1] != actions.shape[1] + 1:
            raise ValueError("Require T issued actions and T+1 observations")
        if actions.shape[1] < self.history:
            raise ValueError("Training clips must contain a full history window")
        pairs = torch.cat([pr[:, :-1], actions], -1)
        # Exclude artificial clip-start padding from the loss. Prefix positions
        # are burn-in context, not supervised transitions. No future p or a leaks.
        windows = torch.stack([
            pairs[:, i-self.history+1:i+1]
            for i in range(self.history-1, actions.shape[1])
        ], dim=1)
        return {
            "pred_body": self.predict_window(windows),
            "target_body": pr[:, self.history:],
            "first_prediction_action_index": self.history-1,
        }


def history_loss(model, batch):
    """Training loss on observable body state, without annotations or goals."""
    out = model(batch)
    loss = F.mse_loss(out["pred_body"], out["target_body"])
    return loss, {
        "loss": float(loss.detach()), "body_prediction": float(loss.detach()),
        "valid_transitions_per_clip": int(out["pred_body"].shape[1]),
    }


def _goal_value(goal, name):
    return goal[name] if isinstance(goal, dict) else getattr(goal, name)


def _push_phase(pr, goal, phase):
    # Contact point and transport endpoint are public task goals. No scene state.
    if phase == 0 and np.linalg.norm(np.asarray(pr[:3]) - np.asarray(_goal_value(goal, "contact_position"))) < .055:
        return 1
    return phase


def _step_cost(pr, goal, phase, device):
    task = _goal_value(goal, "task")
    position = torch.as_tensor(_goal_value(goal, "position"), device=device, dtype=pr.dtype)
    contact = torch.as_tensor(_goal_value(goal, "contact_position"), device=device, dtype=pr.dtype)
    distance = (pr[..., :3] - position).square().sum(-1)
    if task == "grasp_lift":
        held = pr[..., 9].clamp(0, 1)
        approach = (pr[..., :3] - contact).square().sum(-1)
        return (1-held)*approach + held*distance + .08*(1-held)
    if task == "push" and phase == 0:
        return (pr[..., :3] - contact).square().sum(-1)
    if task not in {"reach", "touch", "grasp_lift", "push"}:
        raise ValueError("Unknown physical task: " + task)
    return distance


def _restrict_candidates(candidates, goal):
    candidates[:, :, 4] = 0
    if _goal_value(goal, "task") in {"reach", "push"}:
        candidates[:, :, 3] = -1
    return candidates


def _validate_planner(cfg):
    if cfg.planner_horizon < 1 or cfg.planner_candidates < 2 or cfg.planner_iterations < 1:
        raise ValueError("CEM requires horizon >=1, candidates >=2, and iterations >=1")


class HistoryController:
    """CEM MPC over the learned history predictor; no hidden delay estimate."""
    def __init__(self, model, cfg, device, seed):
        _validate_planner(cfg)
        self.model, self.cfg, self.device = model, cfg, torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.past = deque(maxlen=max(0, model.history-1))
        self.mean = np.zeros((cfg.planner_horizon, ACTION_DIM), np.float32)
        self.phase = 0

    def _context(self, pr, count):
        # At episode reset, repeat the observed initial body state with zero
        # commands. Those are initial padding, not past states from the simulator.
        initial = self.past[0][:PROPRIO_DIM] if self.past else pr
        zero = np.concatenate([initial, np.zeros(ACTION_DIM, np.float32)])
        padding = self.model.history-1-len(self.past)
        arr = np.stack([zero]*padding + list(self.past)) if self.model.history > 1 else np.zeros((0, PROPRIO_DIM+ACTION_DIM), np.float32)
        return torch.as_tensor(arr, device=self.device, dtype=torch.float32)[None].repeat(count, 1, 1)

    @torch.no_grad()
    def rollout(self, obs, candidates):
        """Public diagnostic hook; does not mutate the real action history."""
        actions = torch.as_tensor(candidates, device=self.device, dtype=torch.float32)
        count, horizon = actions.shape[:2]
        pr_np = np.asarray(obs["proprio"], np.float32)
        pr = torch.as_tensor(pr_np, device=self.device)[None].repeat(count, 1)
        context = self._context(pr_np, count)
        predictions = []
        for t in range(horizon):
            current_pair = torch.cat([pr, actions[:, t]], -1).unsqueeze(1)
            window = torch.cat([context, current_pair], 1)
            pr = self.model.predict_window(window)
            predictions.append(pr)
            context = window[:, 1:]
        return torch.stack(predictions, 1)

    @torch.no_grad()
    def __call__(self, obs, goal):
        cfg = self.cfg
        pr = np.asarray(obs["proprio"], np.float32)
        self.phase = _push_phase(pr, goal, self.phase) if _goal_value(goal, "task") == "push" else 0
        mean = self.mean.copy()
        std = np.ones_like(mean)*.75
        for _ in range(cfg.planner_iterations):
            candidates = np.clip(mean[None] + self.rng.normal(size=(cfg.planner_candidates, *mean.shape))*std[None], -1, 1).astype(np.float32)
            candidates[0] = mean
            _restrict_candidates(candidates, goal)
            predictions = self.rollout(obs, candidates)
            cost = torch.zeros(len(candidates), device=self.device)
            for t in range(cfg.planner_horizon):
                cost += _step_cost(predictions[:, t], goal, self.phase, self.device)*(3 if t == cfg.planner_horizon-1 else 1)
            action = torch.as_tensor(candidates, device=self.device)
            cost += .002*action[:, :, :3].square().mean((1, 2))
            elite = torch.topk(cost, max(2, cfg.planner_candidates//8), largest=False).indices.cpu().numpy()
            mean = candidates[elite].mean(0)
            std = candidates[elite].std(0).clip(.10, .8)
        chosen = mean[0].copy()
        # Pair p_t with the command actually issued at t. The next call receives
        # p_(t+1). Never replace this command with a privileged applied_action.
        if self.model.history > 1:
            self.past.append(np.concatenate([pr, chosen]).astype(np.float32))
        self.mean = np.concatenate([mean[1:], mean[-1:]], 0)
        return chosen

    def state_dict(self):
        return {"rng": copy.deepcopy(self.rng.bit_generator.state), "past": [p.tolist() for p in self.past], "mean": self.mean.tolist(), "phase": self.phase}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.past.clear()
        self.past.extend(np.asarray(p, np.float32) for p in state["past"])
        self.mean = np.asarray(state["mean"], np.float32)
        self.phase = int(state["phase"])


class SupplementalController(LearnedController):
    """Legacy Baby controller unchanged for old tasks; adds explicit push cost."""
    def __init__(self, model, cfg, device, seed):
        _validate_planner(cfg)
        super().__init__(model, cfg, device, seed)
        self.phase = 0

    @torch.no_grad()
    def __call__(self, obs, goal):
        if _goal_value(goal, "task") != "push":
            # Reuse the original computation, including its action regularizer,
            # so the frozen control reference does not acquire an untracked fix.
            return super().__call__(obs, SimpleNamespace(**goal) if isinstance(goal, dict) else goal)
        m, cfg = self.model, self.cfg
        self.phase = _push_phase(obs["proprio"], goal, self.phase)
        h, self.state = m.observe(obs, self.previous, self.device, self.state)
        acoustic = torch.as_tensor(obs["audio"], device=self.device).float()[None]
        if m.cfg.ablation == "no_audio":
            acoustic.zero_()
        hearing = m.hearing(acoustic)
        mean = self.mean.copy()
        std = np.ones_like(mean)*.75
        count = cfg.planner_candidates
        for _ in range(cfg.planner_iterations):
            candidates = np.clip(mean[None] + self.rng.normal(size=(count, *mean.shape))*std[None], -1, 1).astype(np.float32)
            candidates[0] = mean
            _restrict_candidates(candidates, goal)
            action = torch.as_tensor(candidates, device=self.device)
            pr = torch.as_tensor(obs["proprio"], device=self.device).float()[None].repeat(count, 1)
            latent = h.repeat(count, 1)
            cost = torch.zeros(count, device=self.device)
            for t in range(cfg.planner_horizon):
                a = m._actions(action[:, t])
                ha = torch.cat([latent, a], -1)
                visual = m.dynamics(ha)
                pr = pr + m.body(ha)
                cost += _step_cost(pr, goal, self.phase, self.device)*(3 if t == cfg.planner_horizon-1 else 1)
                x = torch.cat([visual, hearing.repeat(count, 1), pr, a], -1).unsqueeze(1)
                y, _ = m._memory(x, latent.unsqueeze(0))
                latent = y[:, 0]
            cost += .002*action[:, :, :3].square().mean((1, 2))
            elite = torch.topk(cost, max(2, count//8), largest=False).indices.cpu().numpy()
            mean = candidates[elite].mean(0)
            std = candidates[elite].std(0).clip(.10, .8)
        chosen = mean[0].copy()
        self.mean = np.concatenate([mean[1:], mean[-1:]], 0)
        self.previous = chosen
        return chosen

    def state_dict(self):
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state), "mean": self.mean.tolist(),
            "previous": self.previous.tolist(), "phase": self.phase,
            "recurrent": None if self.state is None else self.state.detach().cpu().clone(),
        }

    def load_state_dict(self, state):
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.mean = np.asarray(state["mean"], np.float32)
        self.previous = np.asarray(state["previous"], np.float32)
        self.phase = int(state["phase"])
        self.state = None if state["recurrent"] is None else torch.as_tensor(state["recurrent"], device=self.device).clone()


class FeedbackController:
    """Fixed proportional/velocity feedback using no simulator state.

    Gains are specified before evaluation and are not read from the environment.
    This is a transparent non-learning control comparison, not an oracle or an
    assertion of optimality. The same explicit task decomposition is supplied.
    """
    def __init__(self, cfg, seed=0, position_gain=4.0, velocity_gain=1.5):
        self.cfg = cfg
        self.position_gain, self.velocity_gain = position_gain, velocity_gain
        self.phase = 0

    def __call__(self, obs, goal):
        pr = np.asarray(obs["proprio"], np.float32)
        task = _goal_value(goal, "task")
        target = np.asarray(_goal_value(goal, "position"), np.float32)
        if task == "grasp_lift" and pr[9] < .5:
            target = np.asarray(_goal_value(goal, "contact_position"), np.float32)
        if task == "push":
            self.phase = _push_phase(pr, goal, self.phase)
            if not self.phase:
                target = np.asarray(_goal_value(goal, "contact_position"), np.float32)
        action = np.zeros(ACTION_DIM, np.float32)
        action[:3] = np.clip(self.position_gain*(target-pr[:3])-self.velocity_gain*pr[3:6], -1, 1)
        action[3] = 1 if task == "grasp_lift" else -1
        return action

    def state_dict(self):
        return {"phase": self.phase}

    def load_state_dict(self, state):
        self.phase = int(state["phase"])


def controller_factory(kind, model, cfg, device, seed):
    """Construct one controller for one episode; never reuse across scenes."""
    if kind in {"world", "learned", "legacy"}:
        return SupplementalController(model, cfg, device, seed)
    if kind in {"history", "history_model"}:
        return HistoryController(model, cfg, device, seed)
    if kind in {"feedback", "pd_feedback"}:
        return FeedbackController(cfg, seed)
    raise ValueError("Unknown supplemental controller: " + str(kind))
