"""Task resolution for the independent GameBench Craftax task."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parents[1]


def stable_hash(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


@dataclass(frozen=True)
class ResolvedTask:
    task_id: str
    scenario_id: str
    seed: int
    width: int
    height: int
    max_steps: int
    world: dict[str, Any]
    rules: dict[str, Any]
    readouts: dict[str, Any]
    config_hash: str
    episode_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "max_steps": self.max_steps,
            "world": copy.deepcopy(self.world),
            "rules": copy.deepcopy(self.rules),
            "readouts": copy.deepcopy(self.readouts),
            "config_hash": self.config_hash,
            "episode_id": self.episode_id,
        }


def load_task_path(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text())


def resolve_task(
    task: dict[str, Any], seed_override: int | None = None
) -> ResolvedTask:
    if str(task.get("schema", "")) not in {"", "gamebench.task.craftax.v1"}:
        raise ValueError(f"unsupported Craftax task schema: {task.get('schema')}")
    task_id = str(task.get("task_id") or task.get("scenario_id") or "manual")
    scenario_id = str(task.get("scenario_id") or task_id)
    raw_world = task.get("world") or {}
    world = _merge_world(raw_world)
    rules = _merge_rules(task.get("rules") or {})
    readouts = _merge_readouts(task.get("readouts") or {})
    if "max_steps" in rules:
        rules["max_steps"] = _strict_positive_int(rules["max_steps"], "rules.max_steps")
    task_max_steps = (
        _strict_positive_int(task["max_steps"], "max_steps")
        if "max_steps" in task
        else 200
    )
    seed = _strict_int(
        seed_override
        if seed_override is not None
        else world.get("seed", task.get("seed", 0)),
        "seed",
    )
    world["seed"] = seed
    width = _strict_int(world.get("width", 48), "world.width")
    height = _strict_int(world.get("height", 48), "world.height")
    levels = _strict_int(world.get("levels", 9), "world.levels")
    max_steps = _strict_positive_int(
        world.get("max_steps", rules.get("max_steps", task_max_steps)), "max_steps"
    )
    if width < 5 or height < 5:
        raise ValueError("Craftax world must be at least 5x5")
    if levels <= 0:
        raise ValueError("Craftax world.levels must be positive")
    world["width"] = width
    world["height"] = height
    world["levels"] = levels
    world["max_steps"] = max_steps
    _canonicalize_nonnegative_int(world, "view_radius", "world.view_radius")
    _canonicalize_nonnegative_int(
        world, "checkpoint_every_n_steps", "world.checkpoint_every_n_steps"
    )
    _canonicalize_positive_int(world, "day_length", "world.day_length")
    _canonicalize_nonnegative_int(
        world, "max_player_projectiles", "world.max_player_projectiles"
    )
    _canonicalize_nonnegative_int(
        world, "max_mob_projectiles", "world.max_mob_projectiles"
    )
    _canonicalize_nonnegative_int(world, "max_passive_mobs", "world.max_passive_mobs")
    _canonicalize_nonnegative_int(world, "max_melee_mobs", "world.max_melee_mobs")
    _canonicalize_nonnegative_int(world, "max_ranged_mobs", "world.max_ranged_mobs")
    _canonicalize_nonnegative_int(
        world, "mob_despawn_distance", "world.mob_despawn_distance"
    )
    _canonicalize_positive_int(rules, "day_length", "rules.day_length")
    _canonicalize_finite_number(rules, "achievement_reward", "rules.achievement_reward")
    _canonicalize_finite_number(rules, "step_reward", "rules.step_reward")
    _canonicalize_finite_number(
        rules, "invalid_action_penalty", "rules.invalid_action_penalty"
    )
    _canonicalize_finite_number(rules, "death_penalty", "rules.death_penalty")
    _canonicalize_bool(rules, "homeostasis", "rules.homeostasis")
    _canonicalize_bool(rules, "god_mode", "rules.god_mode")
    expanded = {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "world": world,
        "rules": rules,
        "readouts": readouts,
    }
    config_hash = stable_hash(expanded, 16)
    episode_id = stable_hash(
        f"gamebench.craftax-singleplayer.episode:{task_id}:{seed}:{config_hash}", 32
    )
    return ResolvedTask(
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
        width=width,
        height=height,
        max_steps=max_steps,
        world=world,
        rules=rules,
        readouts=readouts,
        config_hash=config_hash,
        episode_id=episode_id,
    )


def _merge_world(raw: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    default_name = raw.get("use_default")
    if default_name:
        defaults = _load_json(TASK_DIR / "defaults" / "worlds.json")
        if default_name not in defaults:
            raise ValueError(f"unknown Craftax world default: {default_name}")
        base = copy.deepcopy(defaults[default_name])
    merged = _deep_merge(
        base, {key: value for key, value in raw.items() if key != "use_default"}
    )
    merged.setdefault("width", 48)
    merged.setdefault("height", 48)
    merged.setdefault("view_radius", 4)
    merged.setdefault("levels", 9)
    merged.setdefault("densities", {})
    return merged


def _merge_rules(raw: dict[str, Any]) -> dict[str, Any]:
    base: dict[str, Any] = {}
    base_name = raw.get("base")
    if base_name:
        path = TASK_DIR / "defaults" / "rules" / f"{base_name}.json"
        if path.exists():
            base = _load_json(path)
        else:
            base = {"base": base_name}
    else:
        base = _load_json(TASK_DIR / "defaults" / "rules" / "symbolic_survival.json")
    overrides = raw.get("overrides") or {}
    merged = _deep_merge(
        base,
        {key: value for key, value in raw.items() if key not in {"base", "overrides"}},
    )
    merged = _deep_merge(merged, overrides)
    merged.setdefault("achievement_reward", 1.0)
    merged.setdefault("step_reward", 0.0)
    merged.setdefault("invalid_action_penalty", -0.05)
    merged.setdefault("death_penalty", -1.0)
    merged.setdefault("homeostasis", True)
    return merged


def _merge_readouts(raw: dict[str, Any]) -> dict[str, Any]:
    base = {
        "symbolic": True,
        "local_map": True,
        "full_world_state": False,
        "observation_text": True,
    }
    profile = raw.get("profile")
    if profile:
        defaults = _load_json(TASK_DIR / "defaults" / "readouts.json")
        base = _deep_merge(base, defaults.get(profile, {}))
    return _deep_merge(
        base, {key: value for key, value in raw.items() if key != "profile"}
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"Craftax {field} must be integer: {value!r}")


def _strict_positive_int(value: Any, field: str) -> int:
    number = _strict_int(value, field)
    if number <= 0:
        raise ValueError(f"Craftax {field} must be positive: {value!r}")
    return number


def _strict_nonnegative_int(value: Any, field: str) -> int:
    number = _strict_int(value, field)
    if number < 0:
        raise ValueError(f"Craftax {field} must be nonnegative: {value!r}")
    return number


def _strict_finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return float(value)
    raise ValueError(f"Craftax {field} must be finite number: {value!r}")


def _strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"Craftax {field} must be boolean: {value!r}")


def _canonicalize_positive_int(mapping: dict[str, Any], key: str, field: str) -> None:
    if key in mapping:
        mapping[key] = _strict_positive_int(mapping[key], field)


def _canonicalize_nonnegative_int(
    mapping: dict[str, Any], key: str, field: str
) -> None:
    if key in mapping:
        mapping[key] = _strict_nonnegative_int(mapping[key], field)


def _canonicalize_finite_number(mapping: dict[str, Any], key: str, field: str) -> None:
    if key in mapping:
        mapping[key] = _strict_finite_number(mapping[key], field)


def _canonicalize_bool(mapping: dict[str, Any], key: str, field: str) -> None:
    if key in mapping:
        mapping[key] = _strict_bool(mapping[key], field)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
