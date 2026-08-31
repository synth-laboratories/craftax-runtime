"""Scoring helpers for GameBench Craftax code-policy sweeps."""

from __future__ import annotations


try:
    from parity import CRAFTAX_ACHIEVEMENTS
except ImportError:
    CRAFTAX_ACHIEVEMENTS = []


ACHIEVEMENT_UNIVERSE = list(CRAFTAX_ACHIEVEMENTS)


def achievement_success_score(
    achievement_frequency: dict[str, int], episode_count: int
) -> float:
    if episode_count <= 0:
        return 0.0
    if not ACHIEVEMENT_UNIVERSE:
        raise RuntimeError(
            "Craftax achievement universe is empty; ensure gold_python is on PYTHONPATH"
        )
    total = 0.0
    for name in ACHIEVEMENT_UNIVERSE:
        total += min(1.0, float(achievement_frequency.get(name, 0)) / episode_count)
    return total / len(ACHIEVEMENT_UNIVERSE)


def unique_achievement_reward(achievements: set[str]) -> float:
    return float(len(achievements))
