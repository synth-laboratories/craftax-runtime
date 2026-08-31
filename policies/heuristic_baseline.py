"""Reference Craftax code-policy candidate for the symbolic GameBench task."""

from __future__ import annotations

from collections import deque
from typing import Any


_DIRS = (("right", 1, 0), ("down", 0, 1), ("left", -1, 0), ("up", 0, -1))
_PASSABLE = {".", ">", "<", "P"}


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
    obs = readout["observation"]
    inv = obs["inventory"]
    front = obs["player"]["front_tile"]
    achievements = set(obs.get("achievements", []))
    local_map = obs["local_map"]
    table_near = _nearby(local_map, "A")
    table_known = table_near or "place_table" in achievements

    if front in {"tree", "fire_tree", "ice_shrub"}:
        return _decision("do", "harvest adjacent wood")
    if front in {"stone", "stalagmite"} and inv.get("pickaxe", 0) >= 1:
        return _decision("do", "mine adjacent stone")
    if front == "tree":
        return _decision("do", "harvest adjacent tree")
    if front == "coal" and inv.get("pickaxe", 0) >= 1:
        return _decision("do", "mine adjacent coal")
    if front == "iron" and inv.get("pickaxe", 0) >= 2:
        return _decision("do", "mine adjacent iron")

    if inv.get("pickaxe", 0) < 1:
        if table_near and inv.get("wood", 0) >= 1 and "make_wood_pickaxe" in valid_actions:
            return _decision("make_wood_pickaxe", "craft wood pickaxe at table")
        if table_known and not table_near and inv.get("wood", 0) >= 1:
            for action in _toward_targets(readout, "A"):
                if action in valid_actions:
                    return _decision(action, "return to table for wood pickaxe")
        if not table_known and inv.get("wood", 0) >= 3 and _front_placeable(front) and "place_table" in valid_actions:
            return _decision("place_table", "place crafting table for pickaxe")
        if inv.get("wood", 0) < (1 if table_known else 3):
            for action in _toward_targets(readout, "T"):
                if action in valid_actions:
                    return _decision(action, "seek enough wood for pickaxe")
        if not table_known:
            for action in _toward_open_front(local_map):
                if action in valid_actions:
                    return _decision(action, "find open table placement")
        if table_known:
            for action in _toward_targets(readout, "A"):
                if action in valid_actions:
                    return _decision(action, "return to known table")

    if inv.get("pickaxe", 0) == 1:
        if table_near and inv.get("wood", 0) >= 1 and inv.get("stone", 0) >= 1 and "make_stone_pickaxe" in valid_actions:
            return _decision("make_stone_pickaxe", "upgrade to stone pickaxe")
        if inv.get("stone", 0) < 1:
            for action in _toward_targets(readout, "S^"):
                if action in valid_actions:
                    return _decision(action, "seek stone for stone pickaxe")
        if inv.get("wood", 0) < 1:
            for action in _toward_targets(readout, "T"):
                if action in valid_actions:
                    return _decision(action, "seek wood for stone pickaxe")
        if not table_near and inv.get("wood", 0) >= 1 and inv.get("stone", 0) >= 1:
            for action in _toward_targets(readout, "A"):
                if action in valid_actions:
                    return _decision(action, "return to table for stone pickaxe")

    if inv.get("sapling", 0) >= 1 and "place_plant" not in achievements and _can_take(engine, "place_plant"):
        return _decision("place_plant", "plant collected sapling")
    if inv.get("sapling", 0) >= 1 and "place_plant" not in achievements:
        setup_action = _setup_for_action(engine, valid_actions, "place_plant")
        if setup_action:
            return _decision(setup_action, "set up a valid plant placement")
    if inv.get("stone", 0) >= 1 and "place_furnace" not in achievements and _can_take(engine, "place_furnace"):
        return _decision("place_furnace", "place furnace")
    if inv.get("wood", 0) >= 1 and inv.get("stone", 0) >= 1 and inv.get("sword", 0) < 2 and _can_take(engine, "make_stone_sword"):
        return _decision("make_stone_sword", "make starter sword")
    if front == "grass" and "collect_sapling" not in achievements:
        return _decision("do", "roll grass for sapling")
    for action in _toward_interesting_tile(local_map):
        if action in valid_actions:
            if engine is None or not _is_immediate_violation(engine, action):
                return _decision(action, "move toward nearby resource")
    return _decision(_safe_explore_action(engine, valid_actions), "explore")


def _decision(action: str, reason: str) -> dict[str, Any]:
    return {"actions": [action], "policy_reason": reason}


def _nearby(local_map: list[str], chars: str) -> bool:
    center_y = len(local_map) // 2
    center_x = len(local_map[0]) // 2 if local_map else 0
    for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
        y = center_y + dy
        x = center_x + dx
        if 0 <= y < len(local_map) and 0 <= x < len(local_map[y]) and local_map[y][x] in chars:
            return True
    return False


def _front_placeable(front_tile: str) -> bool:
    return front_tile in {"grass", "sand", "path", "floor", "dirt"}


def _toward_targets(readout: dict[str, Any], targets: str) -> list[str]:
    obs = readout["observation"]
    pos = obs["player"]["pos"]
    lines = str(readout.get("ascii", "")).splitlines()
    if lines and isinstance(pos, list) and len(pos) == 2:
        routed = _route_to_adjacent(lines, (int(pos[0]), int(pos[1])), targets)
        if routed:
            return routed
    return _toward_chars(obs["local_map"], targets) or ["right", "down", "left", "up"]


def _toward_open_front(local_map: list[str]) -> list[str]:
    return _toward_chars(local_map, ".") or ["right", "down", "left", "up"]


def _route_to_adjacent(lines: list[str], start: tuple[int, int], targets: str) -> list[str]:
    facing = _adjacent_direction(lines, start, targets)
    if facing:
        return [facing]
    queue: deque[tuple[int, int]] = deque([start])
    first_action: dict[tuple[int, int], str | None] = {start: None}
    while queue:
        x, y = queue.popleft()
        for action, dx, dy in _DIRS:
            nxt = (x + dx, y + dy)
            if nxt in first_action or not _is_passable(lines, nxt):
                continue
            first_action[nxt] = action if (x, y) == start else first_action[(x, y)]
            if _adjacent_direction(lines, nxt, targets):
                first = first_action[nxt]
                return [first] if first else []
            queue.append(nxt)
    return []


def _adjacent_direction(lines: list[str], pos: tuple[int, int], targets: str) -> str | None:
    x, y = pos
    for action, dx, dy in _DIRS:
        if _char_at(lines, x + dx, y + dy) in targets:
            return action
    return None


def _is_passable(lines: list[str], pos: tuple[int, int]) -> bool:
    x, y = pos
    return _char_at(lines, x, y) in _PASSABLE


def _char_at(lines: list[str], x: int, y: int) -> str:
    if y < 0 or y >= len(lines) or x < 0 or x >= len(lines[y]):
        return ""
    return lines[y][x]


def _toward_interesting_tile(local_map: list[str]) -> list[str]:
    targets = "TASCIHDsr>"
    return _toward_chars(local_map, targets) or ["right", "down", "left", "up"]


def _toward_chars(local_map: list[str], targets: str) -> list[str]:
    center_y = len(local_map) // 2
    center_x = len(local_map[0]) // 2 if local_map else 0
    best: tuple[int, int, int] | None = None
    for y, row in enumerate(local_map):
        for x, char in enumerate(row):
            if char in targets:
                dist = abs(x - center_x) + abs(y - center_y)
                if best is None or dist < best[0]:
                    best = (dist, x - center_x, y - center_y)
    if best is None:
        return []
    _, dx, dy = best
    if abs(dx) >= abs(dy) and dx != 0:
        return ["right" if dx > 0 else "left"]
    if dy != 0:
        return ["down" if dy > 0 else "up"]
    return ["do"]


def _best_immediate_progress(engine: Any, valid_actions: list[str]) -> str | None:
    best: tuple[float, str] | None = None
    for action in valid_actions:
        if action == "noop":
            continue
        sim = engine.clone_for_sim()
        before_invalid = sim.private.invalid_action_count
        sim.step(action)
        if sim.private.invalid_action_count > before_invalid:
            continue
        reward = float(sim.private.reward_last)
        if reward > 0 and (best is None or reward > best[0]):
            best = (reward, action)
    return best[1] if best else None


def _is_immediate_violation(engine: Any, action: str) -> bool:
    sim = engine.clone_for_sim()
    before_invalid = sim.private.invalid_action_count
    sim.step(action)
    return sim.private.invalid_action_count > before_invalid


def _can_take(engine: Any, action: str) -> bool:
    return engine is None or not _is_immediate_violation(engine, action)


def _setup_for_action(engine: Any, valid_actions: list[str], target_action: str) -> str | None:
    if engine is None:
        return None
    for action in ("right", "down", "left", "up"):
        if action not in valid_actions:
            continue
        sim = engine.clone_for_sim()
        before_invalid = sim.private.invalid_action_count
        sim.step(action)
        if sim.private.invalid_action_count > before_invalid:
            continue
        probe = sim.clone_for_sim()
        before_probe_invalid = probe.private.invalid_action_count
        probe.step(target_action)
        if probe.private.invalid_action_count == before_probe_invalid and float(probe.private.reward_last) > 0:
            return action
    return None


def _safe_explore_action(engine: Any, valid_actions: list[str]) -> str:
    for action in ("right", "down", "left", "up", "noop"):
        if action not in valid_actions:
            continue
        if engine is None or not _is_immediate_violation(engine, action):
            return action
    return "noop"
