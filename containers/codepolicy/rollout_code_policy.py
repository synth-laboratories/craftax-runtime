"""Run Craftax code policies in-process against the gold Python engine."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping


TASK_ROOT = Path(__file__).resolve().parents[2]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))
for extra in (TASK_ROOT / "gold_python", TASK_ROOT / "shared", TASK_ROOT / "policies"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from engine import CraftaxEngine  # noqa: E402
from task_resolve import resolve_task  # noqa: E402


PolicyFn = Callable[..., dict[str, Any]]
_POLICY_CACHE: dict[tuple[str, str], PolicyFn] = {}


def policy_stop_reason(decision: Mapping[str, Any]) -> str | None:
    """Return a bounded reason only for an explicit policy stop request.

    Policies may end an episode when continuing would no longer execute the
    candidate being evaluated (for example, after an LLM budget is exhausted).
    Requiring the literal boolean keeps unrelated truthy metadata from stopping
    a benchmark accidentally.
    """

    if decision.get("stop_episode") is not True:
        return None
    reason = decision.get("stop_reason") or decision.get("rationale") or "policy requested stop"
    return str(reason).strip()[:240] or "policy requested stop"


def load_policy_module(policy_path: Path, *, entry: str = "choose_actions") -> PolicyFn:
    resolved = policy_path.expanduser().resolve()
    cache_key = (str(resolved), entry)
    if cache_key in _POLICY_CACHE:
        return _POLICY_CACHE[cache_key]
    if not resolved.is_file():
        raise ValueError(f"policy file not found: {resolved}")
    module_name = (
        f"craftax_codepolicy_{resolved.stem}_{abs(hash(resolved)) % 10_000_000}"
    )
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load policy module: {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, entry, None)
    if fn is None or not callable(fn):
        raise ValueError(f"policy module {resolved} missing callable {entry}")
    _POLICY_CACHE[cache_key] = fn
    return fn


def compile_check_policy(
    policy_path: Path, *, entry: str = "choose_actions"
) -> dict[str, Any]:
    fn = load_policy_module(policy_path, entry=entry)
    task = _load_task("tasks/policy_dev_template.json", seed=1)
    engine = CraftaxEngine()
    engine.reset(resolve_task(task))
    readout = engine.symbolic_readout()
    result = fn(
        observation_text=readout["observation_text"],
        session={"rollout_id": engine.private.episode_id, "ply": 0},
        valid_actions=engine.valid_actions(),
        engine=engine.clone_for_sim(),
        seed=1,
        ply=0,
        readout=readout,
    )
    if not isinstance(result, dict) or not isinstance(result.get("actions"), list):
        raise ValueError("choose_actions must return {'actions': [...]}")
    return {
        "policy_path": str(policy_path.expanduser().resolve()),
        "entry": entry,
        "sample_action": str(result["actions"][0]),
    }


def rollout_code_policy(
    *,
    policy_path: Path,
    seed: int,
    task_path: str = "tasks/policy_dev_template.json",
    max_steps: int = 80,
    policy_entry: str = "choose_actions",
    include_trace: bool = True,
    candidate_fn: PolicyFn | None = None,
    policy_engine_sim: bool = False,
) -> dict[str, Any]:
    task = _load_task(task_path, seed=seed)
    engine = CraftaxEngine()
    engine.reset(resolve_task(task, seed_override=seed))
    candidate = candidate_fn or load_policy_module(policy_path, entry=policy_entry)
    turns: list[dict[str, Any]] = []
    session: dict[str, Any] = {"rollout_id": engine.private.episode_id, "ply": 0}
    ply = 0
    requested_stop: str | None = None
    # A policy that spends the whole episode bumping a wall scores exactly like
    # one that explores and finds nothing: both end with zero achievements. A
    # blocked move is a legal action that no-ops, so `invalid_action_count`
    # stays 0 and nothing surfaces it. Counting the distinct cells the player
    # actually stood on separates "stuck" from "bad".
    visited_cells: set[tuple[int, int, int]] = {
        (engine.world.player_level, *engine.world.player_pos)
    }
    while (
        not engine.private.terminated
        and not engine.private.truncated
        and ply < max_steps
    ):
        readout = engine.symbolic_readout()
        session["ply"] = ply
        sim_engine = engine.clone_for_sim() if policy_engine_sim else None
        decision = candidate(
            observation_text=readout["observation_text"],
            session=session,
            valid_actions=engine.valid_actions(),
            engine=sim_engine,
            seed=seed,
            ply=ply,
            readout=readout,
        )
        requested_stop = policy_stop_reason(decision)
        if requested_stop is not None:
            break
        raw_actions = decision.get("actions") or ["noop"]
        action = str(raw_actions[0])
        engine.step(action)
        visited_cells.add((engine.world.player_level, *engine.world.player_pos))
        if include_trace:
            turns.append(
                {
                    "ply": ply,
                    "action": action,
                    "policy_reason": decision.get("policy_reason", ""),
                    "reward_last": engine.private.reward_last,
                    "reward_total": engine.private.total_reward,
                    "achievements": sorted(engine.private.achievements),
                    "grid_hash": readout["grid_hash"],
                }
            )
        ply += 1
    outcome = (
        "success"
        if engine.private.achievements
        else "truncated"
        if engine.private.truncated or requested_stop is not None
        else "failure"
    )
    result: dict[str, Any] = {
        "trace_correlation_id": f"craftax-codepolicy-{seed}",
        "rollout_id": engine.private.episode_id,
        "success_status": "success",
        "status_detail": outcome,
        "reward_info": {
            "outcome_reward": float(engine.private.total_reward),
            "details": {
                "seed": seed,
                "task_id": engine.resolved.task_id if engine.resolved else "unknown",
                "outcome": outcome,
                "steps": engine.private.step_index,
                "invalid_action_count": engine.private.invalid_action_count,
                "achievement_count": len(engine.private.achievements),
                "achievements": sorted(engine.private.achievements),
                "policy_path": str(policy_path.expanduser().resolve()),
                "policy_requested_stop": requested_stop is not None,
                "policy_stop_reason": requested_stop,
                "distinct_player_cells": len(visited_cells),
            },
        },
        "state": {
            "public": engine.public.to_dict(),
            "private": engine.private.to_dict(),
        },
        "artifact": [{"artifact_type": "turns", "turns": turns}],
    }
    if include_trace:
        result["events"] = engine.nev.legacy_strings()
        result["nev"] = engine.nev.export()
    return result


def _load_task(task_path: str, *, seed: int) -> dict[str, Any]:
    task = json.loads((TASK_ROOT / task_path).read_text())
    task["seed"] = seed
    task["task_id"] = f"{task.get('task_id', 'craftax_policy_dev')}_{seed}"
    return task
