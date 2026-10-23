from dataclasses import dataclass, asdict
import json
from pathlib import Path

@dataclass
class Config:
    seed: int = 0
    image_size: int = 64
    control_hz: int = 8
    physics_hz: int = 120
    sample_rate: int = 16000
    episode_steps: int = 64
    objects_min: int = 1
    objects_max: int = 3
    correspondence: float = 1.0
    caregiver_delay_steps: int = 0
    exploration: str = "random"
    ablation: str = "none"
    hidden: int = 128
    batch_size: int = 16
    clip_steps: int = 16
    learning_rate: float = 0.0003
    blocks: int = 8
    episodes_per_block: int = 16
    updates_per_block: int = 150
    eval_per_condition: int = 48
    eval_namespace: str = "development-v1"
    eval_interval_blocks: int = 1
    evaluation_mode: str = "multimodal"
    motor_domain_randomization: bool = False
    physics_trials: int = 6
    physics_steps: int = 40
    planner_horizon: int = 8
    planner_candidates: int = 48
    planner_iterations: int = 2
    eval_dynamics_episodes: int = 4
    motor_trials: int = 8
    motor_steps: int = 24
    replay_episodes: int = 512
    probe_examples: int = 160
    threads: int = 4
    max_seconds: int = 7200
    save_video: bool = True

    def validate(self):
        assert 0 <= self.correspondence <= 1
        assert self.image_size == 64, "This baseline uses 64x64 rendered observations."
        assert self.physics_hz % self.control_hz == 0
        assert self.sample_rate % self.control_hz == 0
        assert self.episode_steps >= self.clip_steps + 1
        assert self.batch_size >= 2
        assert 1 <= self.objects_min <= self.objects_max <= 6
        assert self.exploration in {"random", "learning_progress"}
        assert self.ablation in {"none", "no_audio", "no_action", "no_world_memory", "shuffle_audio"}
        assert self.caregiver_delay_steps >= 0
        assert self.eval_namespace and self.eval_interval_blocks >= 1
        assert self.evaluation_mode in {"multimodal", "physics"}
        assert self.physics_trials >= 1 and self.physics_steps >= 3
        assert self.planner_horizon >= 1 and self.planner_candidates >= 4 and self.planner_iterations >= 1
        return self

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
