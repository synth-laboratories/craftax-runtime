"""Scalar GEPA reward from unique Craftax achievements."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

TASK_ROOT = Path(__file__).resolve().parents[2]
for extra in (TASK_ROOT, TASK_ROOT / "gold_python", TASK_ROOT / "shared"):
    path = str(extra)
    if path not in sys.path:
        sys.path.insert(0, path)

from scoring import ACHIEVEMENT_UNIVERSE, achievement_success_score  # noqa: E402


def episode_reward(achievements: Iterable[str]) -> float:
    unique = {str(name) for name in achievements}
    freq = {name: 1 for name in unique}
    return float(achievement_success_score(freq, 1))


def universe_size() -> int:
    return len(ACHIEVEMENT_UNIVERSE)
