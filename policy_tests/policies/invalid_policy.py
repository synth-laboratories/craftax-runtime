"""Intentionally invalid Craftax policy used by policy contract tests."""

from __future__ import annotations

from typing import Any


def choose_actions(
    *,
    observation_text: str,
    session: dict[str, Any],
    valid_actions: list[str],
    engine: Any = None,
    readout: dict[str, Any],
    seed: int,
    ply: int,
) -> dict[str, Any]:
    return {"actions": ["dig_sideways"], "policy_reason": "invalid action smoke"}

