"""Public/private state split for GameBench Craftax."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PublicState:
    observation: dict[str, Any]
    player_pos: tuple[int, int]
    level: int
    inventory: dict[str, Any]
    achievements: dict[str, int]
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": copy.deepcopy(self.observation),
            "player_pos": [self.player_pos[0], self.player_pos[1]],
            "level": self.level,
            "inventory": copy.deepcopy(self.inventory),
            "achievements": dict(self.achievements),
            "done": self.done,
        }


@dataclass
class PrivateState:
    episode_id: str = ""
    task_id: str = ""
    scenario_id: str = ""
    seed: int = 0
    config_hash: str = ""
    step_index: int = 0
    reward_last: float = 0.0
    total_reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    done_reason: str | None = None
    achievements: set[str] = field(default_factory=set)
    invalid_action_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "step_index": self.step_index,
            "reward_last": self.reward_last,
            "total_reward": self.total_reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done_reason": self.done_reason,
            "achievements": sorted(self.achievements),
            "invalid_action_count": self.invalid_action_count,
        }
