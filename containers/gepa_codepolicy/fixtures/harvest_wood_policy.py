"""Harvest wood when adjacent; otherwise walk toward trees."""

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
    obs = readout.get("observation") or {}
    front = (obs.get("player") or {}).get("front_tile")
    if front in {"tree", "fire_tree", "ice_shrub"} and "do" in valid_actions:
        return {"actions": ["do"], "policy_reason": "harvest wood"}
    local_map = obs.get("local_map") or []
    for action in _toward(local_map, "T"):
        if action in valid_actions:
            return {"actions": [action], "policy_reason": "seek tree"}
    action = "noop" if "noop" in valid_actions else (valid_actions[0] if valid_actions else "noop")
    return {"actions": [action], "policy_reason": "idle"}


def _toward(local_map: list[str], targets: str) -> list[str]:
    if not local_map or not local_map[0]:
        return []
    cy = len(local_map) // 2
    cx = len(local_map[0]) // 2
    best: tuple[int, int, int] | None = None
    for y, row in enumerate(local_map):
        for x, char in enumerate(row):
            if char in targets:
                dist = abs(x - cx) + abs(y - cy)
                if best is None or dist < best[0]:
                    best = (dist, x - cx, y - cy)
    if best is None:
        return []
    _, dx, dy = best
    if abs(dx) >= abs(dy) and dx != 0:
        return ["right" if dx > 0 else "left"]
    if dy != 0:
        return ["down" if dy > 0 else "up"]
    return ["do"]
