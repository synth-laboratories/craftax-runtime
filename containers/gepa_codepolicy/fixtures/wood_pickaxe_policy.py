"""Wood → table → wood pickaxe. No engine.clone_for_sim()."""

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
    inv = obs.get("inventory") or {}
    front = (obs.get("player") or {}).get("front_tile")
    achievements = set(obs.get("achievements") or [])
    local_map = obs.get("local_map") or []
    wood = int(inv.get("wood") or 0)
    pickaxe = int(inv.get("pickaxe") or 0)
    table_near = _nearby(local_map, "Aa")

    if table_near and pickaxe < 1 and wood >= 1 and "make_wood_pickaxe" in valid_actions:
        return {"actions": ["make_wood_pickaxe"], "policy_reason": "craft pickaxe at table"}
    if "place_table" not in achievements and wood >= 3 and "place_table" in valid_actions:
        return {"actions": ["place_table"], "policy_reason": "place table"}
    if front in {"tree", "fire_tree", "ice_shrub"} and "do" in valid_actions:
        if pickaxe < 1 and wood < (1 if "place_table" in achievements else 3):
            return {"actions": ["do"], "policy_reason": "chop wood"}
    if pickaxe < 1 and wood < (1 if "place_table" in achievements else 3):
        for action in _toward(local_map, "T"):
            if action in valid_actions:
                return {"actions": [action], "policy_reason": "seek tree"}
    if pickaxe < 1 and "place_table" in achievements and not table_near:
        for action in _toward(local_map, "Aa"):
            if action in valid_actions:
                return {"actions": [action], "policy_reason": "return to table"}
    if pickaxe < 1 and wood >= 3 and "place_table" not in achievements:
        for action in ("right", "down", "left", "up"):
            if action in valid_actions:
                return {"actions": [action], "policy_reason": "find grass for table"}
    action = "noop" if "noop" in valid_actions else (valid_actions[0] if valid_actions else "noop")
    return {"actions": [action], "policy_reason": "idle"}


def _nearby(local_map: list[str], chars: str) -> bool:
    if not local_map or not local_map[0]:
        return False
    cy = len(local_map) // 2
    cx = len(local_map[0]) // 2
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        y, x = cy + dy, cx + dx
        if 0 <= y < len(local_map) and 0 <= x < len(local_map[y]) and local_map[y][x] in chars:
            return True
    return False


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
