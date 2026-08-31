"""Always-noop Craftax code policy."""

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
    action = "noop" if "noop" in valid_actions else (valid_actions[0] if valid_actions else "noop")
    return {"actions": [action], "policy_reason": "noop"}
