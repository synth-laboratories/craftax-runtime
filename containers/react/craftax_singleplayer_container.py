"""GameBench Craftax single-player ReAct container (rust gold HTTP)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from containers.react.agent_policy import AgentPolicy, AgentPolicyConfig
from containers.react.craftax_rust_client import (
    CraftaxGoldRequestError,
    CraftaxRustGoldClient,
)
from containers.react.trace_emitter import CraftaxTrace

TASK_DIR = Path(__file__).resolve().parents[2]
TASK_FAMILY = "craftax_singleplayer_rust"
DEFAULT_OBJECTIVE = "collect resources, craft tools, and unlock Craftax achievements"
app = FastAPI(title="gamebench-craftax-singleplayer-container", version="0.1.0")
ASYNC_ROLLOUTS: dict[str, dict[str, Any]] = {}
ROLLOUT_TASKS: dict[str, asyncio.Task[Any]] = {}
CHECKPOINTS: dict[str, dict[str, Any]] = {}
GOLD = CraftaxRustGoldClient()


class RolloutEnvSpec(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None


class RolloutPolicySpec(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class RolloutRequest(BaseModel):
    trace_correlation_id: str | None = None
    trial_id: str | None = None
    trace_context: dict[str, str] | None = None
    env: RolloutEnvSpec = Field(default_factory=RolloutEnvSpec)
    policy: RolloutPolicySpec = Field(default_factory=RolloutPolicySpec)


def _world_identity(task: dict[str, Any]) -> dict[str, Any]:
    """Name the world that was actually played.

    A Craftax score is meaningless without this. The 2026-08-12 Luna run
    reported `reward 0.98` with no record that it was a 16x16, 3-level board
    with 16% of vanilla tree density and a 120-step budget, so the number read
    as a Craftax capability result. Every rollout now carries its world so that
    confusion cannot recur, and `is_reference_world` says in one boolean
    whether the score is comparable to a real Craftax run.
    """
    world = task.get("world") or {}
    densities = world.get("densities") or {}
    width = int(world.get("width", 48))
    height = int(world.get("height", 48))
    levels = int(world.get("levels", 9))
    max_steps = int(task.get("max_steps") or world.get("max_steps") or 0)
    # Vanilla Craftax: full board, all nine levels, unscaled densities.
    is_reference = (
        width >= 48
        and height >= 48
        and levels >= 9
        and not densities
    )
    return {
        "preset": world.get("use_default"),
        "width": width,
        "height": height,
        "levels": levels,
        "max_steps": max_steps,
        "densities": densities,
        "is_reference_world": is_reference,
        "scaled_resources": sorted(densities.keys()),
    }


def _load_task(env_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(env_config.get("task"), dict):
        task = dict(env_config["task"])
    else:
        # Real Craftax (48x48, 9 levels, vanilla densities), never a fixture
        # room. The old default was `policy_dev_template.json` -> `fixture_room`
        # (9x9, 2 levels, tree density 0.0), so any caller that omitted
        # `task_path` silently evaluated a unit-test arena and reported it as a
        # Craftax score. Every real caller passes task_path explicitly; only the
        # forgetful one landed here.
        task_path = str(env_config.get("task_path", "tasks/craftax_default_template.json"))
        path = TASK_DIR / task_path
        if not path.is_file():
            raise HTTPException(
                status_code=400, detail=f"unknown_task_path:{task_path}"
            )
        task = json.loads(path.read_text(encoding="utf-8"))
    if "max_steps" in env_config:
        max_steps = int(env_config["max_steps"])
        task["max_steps"] = max_steps
        world = task.get("world")
        if isinstance(world, dict):
            world["max_steps"] = max_steps
    return task


def _private(readout: dict[str, Any]) -> dict[str, Any]:
    private = readout.get("private")
    if not isinstance(private, dict):
        private = readout.get("public", {}).get("private")
    if not isinstance(private, dict):
        raise RuntimeError("craftax readout missing private object")
    return private


def _state_from_readout(readout: dict[str, Any]) -> dict[str, Any]:
    private = _private(readout)
    reward = float(private.get("total_reward") or 0.0)
    return {
        "terminated": bool(private.get("terminated")),
        "truncated": bool(private.get("truncated")),
        "reward": reward,
        "total_reward": reward,
        "private": private,
        "public": readout.get("public")
        if isinstance(readout.get("public"), dict)
        else {},
    }


def _achievement_names(readout: dict[str, Any]) -> list[str]:
    private = _private(readout)
    raw = private.get("achievements")
    if isinstance(raw, list):
        return sorted(str(item) for item in raw)
    if isinstance(raw, dict):
        return sorted(str(key) for key, value in raw.items() if value)
    public = readout.get("public") if isinstance(readout.get("public"), dict) else {}
    raw_public = public.get("achievements")
    if isinstance(raw_public, dict):
        return sorted(str(key) for key, value in raw_public.items() if value)
    return []


def _emit_achievement_rewards(
    trace: CraftaxTrace,
    *,
    step: int,
    new_achievements: list[str],
    current_achievements: set[str],
    total_reward: float,
    action_event: str | None,
    observation_event: str | None,
) -> list[str]:
    """Emit exactly one sparse reward fact for each newly unlocked achievement."""

    emitted: list[str] = []
    for achievement in new_achievements:
        reward_event = trace.event(
            "environment.reward",
            {
                "step": step,
                "achievement": achievement,
                "value": 1.0,
                "total_reward": total_reward,
                "achievements": sorted(current_achievements),
            },
            caused_by=tuple(
                item for item in (action_event, observation_event) if item
            ),
            structural={"step": step, "achievement": achievement},
            actor="environment",
        )
        if reward_event:
            emitted.append(reward_event)
    return emitted


def _objective_labels(readout: dict[str, Any], task: dict[str, Any]) -> list[str]:
    private = _private(readout)
    labels = [f"objective:{task.get('task_id', 'craftax_progress')}"]
    labels.extend(f"achievement:{name}" for name in _achievement_names(readout))
    if private.get("terminated"):
        labels.append("terminated")
    if private.get("truncated"):
        labels.append("truncated")
    reward = float(private.get("total_reward") or 0.0)
    if reward > 0.0:
        labels.append(f"reward:{reward:.4f}")
    return sorted(set(labels))


def _raise_gold_error(exc: Exception) -> None:
    if isinstance(exc, CraftaxGoldRequestError):
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if isinstance(exc, httpx.HTTPError):
        raise HTTPException(
            status_code=502, detail=f"craftax_gold_http_error:{exc}"
        ) from exc
    raise exc


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        gold = await GOLD.health()
    except Exception as exc:
        _raise_gold_error(exc)
    return {
        "healthy": True,
        "status": "ok",
        "service": "gamebench_craftax_singleplayer_container",
        "gold_lane": gold.get("lane", "rust"),
        "supports_many_live_seeds": True,
    }


@app.get("/info")
async def info() -> dict[str, Any]:
    return {
        "service": {
            "task": "gamebench-craftax-rust-react",
            "adapter": "synth-task-app",
            "authority": "gamebench-craftax-rust",
        },
        "contract": {
            "rollout_request": "synth RolloutRequest {env:{env_name,config,seed}, policy:{...}}",
            "reward": "reward_info.outcome_reward",
            "accounting": [
                "reward_info.details.input_tokens",
                "reward_info.details.llm_calls",
                "reward_info.details.call_accounting",
            ],
        },
        "policy_model": "gemini-3.1-flash-lite",
        "token_budget_per_episode": 20000,
    }


@app.get("/metadata")
async def metadata() -> dict[str, Any]:
    return {
        "status": "ok",
        "runtime_id": "gamebench.craftax_singleplayer.react",
        "name": "GameBench Craftax single-player ReAct container",
        "task_family": TASK_FAMILY,
        "go_ex": {"async_rollouts": True},
        "capabilities": {
            "go_ex": {"async_rollouts": True},
            "async_rollout": True,
            "checkpoint_resume": True,
            "scheduled_checkpoints": True,
        },
        "features": ["async_rollout", "checkpoint_resume", "scheduled_checkpoints"],
    }


@app.get("/task_info")
async def task_info(
    seed: int = 1,
    seeds: list[int] = Query(default=[]),
    task_path: str = "tasks/policy_dev_template.json",
) -> dict[str, Any] | list[dict[str, Any]]:
    task = _load_task({"task_path": task_path})
    seed_list = seeds if seeds else [seed]
    payloads = [
        {
            "seed": item,
            "task_family": TASK_FAMILY,
            "task_id": task.get("task_id"),
            "objective": DEFAULT_OBJECTIVE,
            "reward_mode": "progress",
        }
        for item in seed_list
    ]
    return payloads[0] if len(payloads) == 1 else payloads


@app.post("/rollout")
async def rollout(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="rollout_request_must_be_object")
    if (
        payload.get("submission_mode") == "async"
        or payload.get("schema_version") == "goex_rollout_request.v1"
    ):
        return await _submit_rollout_background(payload)
    parsed = RolloutRequest.model_validate(payload)
    env_config = dict(parsed.env.config)
    seed = int(parsed.env.seed or env_config.get("seed") or 1)
    env_config["seed"] = seed
    record = await _execute_goex_rollout(
        {
            "env": {"config": env_config, "seed": seed},
            "policy": {"config": dict(parsed.policy.config)},
            "trace_correlation_id": parsed.trace_correlation_id
            or f"craftax-react-{seed}",
            "trial_id": parsed.trial_id,
            "trace_context": parsed.trace_context,
        }
    )
    ASYNC_ROLLOUTS[str(record["rollout_id"])] = record
    return record


@app.post("/rollouts")
async def rollouts(request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="rollout_request_must_be_object")
    return await _submit_rollout_background(payload)


@app.post("/rollouts/{parent_rollout_id}/resume")
@app.post("/rollouts/{parent_rollout_id}/resume_async")
async def resume_async(parent_rollout_id: str, request: Request) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="resume_request_must_be_object")
    overrides = payload.get("overrides")
    effective_payload = (
        dict(overrides) if isinstance(overrides, dict) else dict(payload)
    )
    if "trace_context" not in effective_payload and isinstance(
        payload.get("trace_context"),
        dict,
    ):
        effective_payload["trace_context"] = dict(payload["trace_context"])
    checkpoint_id = str(
        payload.get("checkpoint_id")
        or effective_payload.get("checkpoint_id")
        or effective_payload.get("resume_from_checkpoint_id")
        or ""
    ).strip()
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="checkpoint_id_required")
    target_rollout_id = str(
        payload.get("target_rollout_id")
        or effective_payload.get("target_rollout_id")
        or ""
    ).strip()
    if target_rollout_id:
        effective_payload["rollout_id"] = target_rollout_id
    effective_payload.setdefault("checkpoint_id", checkpoint_id)
    effective_payload.setdefault("resume_from_checkpoint_id", checkpoint_id)
    record = await _execute_goex_rollout(
        effective_payload,
        parent_rollout_id=parent_rollout_id,
        checkpoint_id=checkpoint_id,
    )
    rollout_id = str(record["rollout_id"])
    ASYNC_ROLLOUTS[rollout_id] = record
    return {"rollout_id": rollout_id, "status": "submitted", "submission_mode": "async"}


@app.get("/rollouts/{rollout_id}/state")
async def rollout_state(rollout_id: str) -> dict[str, Any]:
    record = ASYNC_ROLLOUTS.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    reward = float(record.get("reward_info", {}).get("outcome_reward", 0.0))
    progress = (
        record.get("progress") if isinstance(record.get("progress"), dict) else {}
    )
    details = record.get("reward_info", {}).get("details")
    details_map = dict(details) if isinstance(details, dict) else {}
    metadata = (
        record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    )
    if not progress and str(record.get("status") or "") in {"completed", "failed"}:
        llm_calls = int(
            metadata.get("policy_llm_turns") or details_map.get("steps") or 0
        )
        progress = {
            "llm_calls": llm_calls,
            "max_llm_calls": llm_calls,
            "steps": int(details_map.get("steps") or llm_calls),
            "max_steps": int(details_map.get("steps") or llm_calls),
            "reward": reward,
            "achievements": list(details_map.get("achievements") or []),
            "new_achievements": [],
        }
    return {
        "rollout_id": rollout_id,
        "status": record.get("status", "completed"),
        "reward": reward,
        "achievements": list(
            progress.get("achievements") or details_map.get("achievements") or []
        ),
        "progress": progress,
        "objective_labels": record.get("objective_labels", []),
        "scheduled_checkpoints": record.get("scheduled_checkpoints", []),
        "metadata": record.get("metadata", {}),
        "reward_info": record.get("reward_info", {}),
        "summary": record.get("summary", {}),
        "error": record.get("error"),
    }


@app.get("/rollouts/{rollout_id}/record")
async def rollout_record(rollout_id: str) -> dict[str, Any]:
    record = ASYNC_ROLLOUTS.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    return record


@app.get("/rollouts/{rollout_id}")
async def get_rollout(rollout_id: str) -> dict[str, Any]:
    return await rollout_record(rollout_id)


@app.post("/rollouts/{rollout_id}/checkpoints")
async def create_checkpoint(rollout_id: str, request: Request) -> dict[str, Any]:
    record = ASYNC_ROLLOUTS.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    payload = await request.json()
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="checkpoint_request_must_be_object")
    checkpoint_id = str(
        payload.get("checkpoint_id") or record.get("terminal_checkpoint_id") or ""
    ).strip()
    if not checkpoint_id:
        raise HTTPException(status_code=400, detail="checkpoint_id_required")
    terminal_id = str(record.get("terminal_checkpoint_id") or "")
    terminal_checkpoint = CHECKPOINTS.get(terminal_id)
    if terminal_checkpoint is None:
        raise HTTPException(status_code=409, detail="rollout_snapshot_unavailable")
    if checkpoint_id != terminal_id:
        checkpoint = {**terminal_checkpoint, "checkpoint_id": checkpoint_id}
        CHECKPOINTS[checkpoint_id] = checkpoint
    else:
        checkpoint = terminal_checkpoint
    return _public_checkpoint(checkpoint)


@app.get("/rollouts/{rollout_id}/checkpoints")
async def list_checkpoints(rollout_id: str) -> dict[str, Any]:
    record = ASYNC_ROLLOUTS.get(rollout_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown_rollout:{rollout_id}")
    checkpoint_ids = [
        str(item.get("checkpoint_id"))
        for item in record.get("scheduled_checkpoints", [])
    ]
    terminal_id = record.get("terminal_checkpoint_id")
    if terminal_id:
        checkpoint_ids.append(str(terminal_id))
    checkpoints = [
        _public_checkpoint(CHECKPOINTS[item])
        for item in checkpoint_ids
        if item in CHECKPOINTS
    ]
    return {"rollout_id": rollout_id, "checkpoints": checkpoints}


@app.get("/rollouts/{rollout_id}/checkpoints/{checkpoint_id}")
async def get_checkpoint(rollout_id: str, checkpoint_id: str) -> dict[str, Any]:
    checkpoint = CHECKPOINTS.get(checkpoint_id)
    if checkpoint is None or checkpoint.get("rollout_id") != rollout_id:
        raise HTTPException(
            status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}"
        )
    return checkpoint


async def _submit_rollout_background(payload: dict[str, Any]) -> dict[str, Any]:
    rollout_id = _requested_rollout_id(payload) or f"craftax-goex-{uuid.uuid4()}"
    payload = {
        **payload,
        "trace_correlation_id": str(payload.get("trace_correlation_id") or rollout_id),
    }
    ASYNC_ROLLOUTS[rollout_id] = _initial_running_record(payload, rollout_id)

    async def _run() -> None:
        try:
            record = await _execute_goex_rollout(
                payload, progress_rollout_id=rollout_id
            )
            ASYNC_ROLLOUTS[rollout_id] = record
        except Exception as exc:
            failed = dict(ASYNC_ROLLOUTS.get(rollout_id) or {})
            failed.update({"status": "failed", "error": str(exc)})
            ASYNC_ROLLOUTS[rollout_id] = failed

    ROLLOUT_TASKS[rollout_id] = asyncio.create_task(_run())
    return {"rollout_id": rollout_id, "status": "submitted", "submission_mode": "async"}


def _initial_running_record(payload: dict[str, Any], rollout_id: str) -> dict[str, Any]:
    env_spec = payload.get("env") if isinstance(payload.get("env"), dict) else {}
    env_config_raw = (
        env_spec.get("config") if isinstance(env_spec.get("config"), dict) else {}
    )
    policy_spec = (
        payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    )
    policy_config_raw = (
        policy_spec.get("config") if isinstance(policy_spec.get("config"), dict) else {}
    )
    seed = int(
        env_spec.get("seed") or env_config_raw.get("seed") or payload.get("seed") or 1
    )
    max_steps = max(int(env_config_raw.get("max_steps", 20)), 1)
    max_llm_turns = max(int(policy_config_raw.get("max_llm_turns", max_steps)), 1)
    metadata = dict(
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    )
    return {
        "rollout_id": rollout_id,
        "trace_correlation_id": str(payload.get("trace_correlation_id") or rollout_id),
        "status": "running",
        "metadata": metadata,
        "reward_info": {
            "outcome_reward": 0.0,
            "details": {"seed": seed, "achievements": []},
        },
        "progress": {
            "llm_calls": 0,
            "max_llm_calls": max_llm_turns,
            "steps": 0,
            "max_steps": max_steps,
            "reward": 0.0,
            "achievements": [],
            "new_achievements": [],
        },
        "artifact": [{"artifact_type": "turns", "turns": []}],
        "artifacts": [{"artifact_type": "turns", "turns": []}],
    }


def _usage_prompt_tokens(usage: Mapping[str, Any]) -> int:
    for key in ("prompt_tokens", "input_tokens", "prompt_token_count"):
        value = usage.get(key)
        if value is not None:
            return int(value)
    return 0


def _usage_completion_tokens(usage: Mapping[str, Any]) -> int:
    for key in ("completion_tokens", "output_tokens", "candidates_token_count"):
        value = usage.get(key)
        if value is not None:
            return int(value)
    return 0


def _turn_usage_totals(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokens where they were reported, and nothing where they were not.

    This counted a call only when its usage payload was non-empty, so a provider
    that returns no usage produced `llm_calls: 0` beside an outer record showing
    three real calls, and every token field read `0` — a number, indistinguishable
    from a call that genuinely used none. Every derived rate was then wrong in a
    way nothing flagged.

    Executed calls come from the call records. Usage-bearing calls are a separate
    coverage figure. Absent tokens are `None`.
    """
    prompt_tokens = 0
    completion_tokens = 0
    executed_calls = 0
    usage_bearing_calls = 0
    for turn in turns:
        # One call can span several plies; the first ply of a batch carries it.
        if turn.get("batch_index") in (0, None):
            executed_calls += 1
        usage = turn.get("usage")
        if not isinstance(usage, Mapping) or not usage:
            continue
        usage_bearing_calls += 1
        prompt_tokens += _usage_prompt_tokens(usage)
        completion_tokens += _usage_completion_tokens(usage)
    reported = usage_bearing_calls > 0
    return {
        "input_tokens": prompt_tokens if reported else None,
        "prompt_tokens": prompt_tokens if reported else None,
        "completion_tokens": completion_tokens if reported else None,
        "total_tokens": prompt_tokens + completion_tokens if reported else None,
        # The count of calls that actually happened, which is what "how many
        # model calls did this rollout make" means.
        "llm_calls": executed_calls,
        "usage_bearing_calls": usage_bearing_calls,
        "usage_coverage": {
            "reported_by": usage_bearing_calls,
            "of": executed_calls,
            "complete": reported and usage_bearing_calls == executed_calls,
        },
    }


def _update_rollout_progress(
    rollout_id: str,
    *,
    llm_calls: int,
    max_llm_turns: int,
    step: int,
    max_steps: int,
    readout: dict[str, Any],
    turns: list[dict[str, Any]],
    seed: int,
    metadata: dict[str, Any],
) -> None:
    record = ASYNC_ROLLOUTS.get(rollout_id)
    if record is None:
        return
    achievements = _achievement_names(readout)
    previous = set(
        str(item) for item in (record.get("progress") or {}).get("achievements") or []
    )
    new_achievements = [name for name in achievements if name not in previous]
    reward = float(_private(readout).get("total_reward") or 0.0)
    usage_totals = _turn_usage_totals(turns)
    progress = {
        "llm_calls": llm_calls,
        "max_llm_calls": max_llm_turns,
        "steps": step,
        "max_steps": max_steps,
        "reward": reward,
        "achievements": achievements,
        "new_achievements": new_achievements,
    }
    record.update(
        {
            "status": "running",
            "metadata": {
                **metadata,
                "policy_llm_turns": llm_calls,
                "achievement_count": len(achievements),
            },
            "reward_info": {
                "outcome_reward": reward,
                "details": {
                    "seed": seed,
                    "achievements": achievements,
                    "steps": step,
                    "input_tokens": usage_totals["input_tokens"],
                    "prompt_tokens": usage_totals["prompt_tokens"],
                    "completion_tokens": usage_totals["completion_tokens"],
                    "llm_calls": llm_calls,
                    "call_accounting": usage_totals,
                    "model": metadata.get("policy_model"),
                },
            },
            "progress": progress,
            "artifact": [{"artifact_type": "turns", "turns": turns}],
            "artifacts": [{"artifact_type": "turns", "turns": turns}],
        }
    )
    ASYNC_ROLLOUTS[rollout_id] = record


async def _execute_goex_rollout(
    request: dict[str, Any],
    *,
    parent_rollout_id: str | None = None,
    checkpoint_id: str | None = None,
    progress_rollout_id: str | None = None,
) -> dict[str, Any]:
    trace = CraftaxTrace.from_request(request)
    env_spec = request.get("env") if isinstance(request.get("env"), dict) else {}
    policy_spec = (
        request.get("policy") if isinstance(request.get("policy"), dict) else {}
    )
    env_config_raw = (
        env_spec.get("config") if isinstance(env_spec.get("config"), dict) else {}
    )
    policy_config_raw = (
        policy_spec.get("config") if isinstance(policy_spec.get("config"), dict) else {}
    )
    seed = int(
        env_spec.get("seed") or env_config_raw.get("seed") or request.get("seed") or 1
    )
    rollout_id = _requested_rollout_id(request) or f"craftax-goex-{uuid.uuid4()}"
    trace_correlation_id = str(request.get("trace_correlation_id") or rollout_id)
    task = _load_task(env_config_raw)
    objective = str(env_config_raw.get("objective") or DEFAULT_OBJECTIVE)
    max_steps = max(int(env_config_raw.get("max_steps", task.get("max_steps", 80))), 1)
    task["max_steps"] = max_steps
    world = task.get("world")
    if isinstance(world, dict):
        world["max_steps"] = max_steps
    agent = AgentPolicy(AgentPolicyConfig.from_mapping(policy_config_raw), trace=trace)
    max_llm_turns = max(int(policy_config_raw.get("max_llm_turns", max_steps)), 1)
    checkpoint_schedule = (
        request.get("checkpoint_schedule")
        if isinstance(request.get("checkpoint_schedule"), dict)
        else {}
    )
    schedule_per_turn = checkpoint_schedule.get("mode") == "per_llm_call"
    checkpoint_prefix = str(
        checkpoint_schedule.get("checkpoint_id_prefix")
        or f"craftax-goex-midcp-{rollout_id}"
    )
    request_metadata = dict(
        request.get("metadata") if isinstance(request.get("metadata"), dict) else {}
    )
    request_metadata.setdefault("policy_model", agent.config.model)

    try:
        trace_refs: dict[str, list[str]] = {
            "observations": [],
            "proposals": [],
            "executed_actions": [],
            "transitions": [],
            "rewards": [],
            "checkpoints": [],
            "lifecycle": [],
        }
        rollout_started_event = trace.event(
            "environment.rollout_started",
            {
                "rollout_id": rollout_id,
                "trace_correlation_id": trace_correlation_id,
                "parent_rollout_id": parent_rollout_id,
                "parent_checkpoint_id": checkpoint_id,
                "task_id": task.get("task_id"),
                "seed": seed,
                "max_steps": max_steps,
            },
            structural={"rollout_id": rollout_id, "seed": seed},
            actor="environment",
        )
        if rollout_started_event:
            trace_refs["lifecycle"].append(rollout_started_event)
        created = await GOLD.create_rollout(task=task, seed=seed)
        gold_rollout_id = str(created.get("rollout_id") or "").strip()
        if not gold_rollout_id:
            raise RuntimeError("craftax gold create_rollout missing rollout_id")

        if checkpoint_id:
            checkpoint = CHECKPOINTS.get(checkpoint_id)
            if checkpoint is None:
                raise HTTPException(
                    status_code=404, detail=f"unknown_checkpoint:{checkpoint_id}"
                )
            await GOLD.restore(gold_rollout_id, str(checkpoint["blob_b64"]))
            restored_event = trace.event(
                "environment.checkpoint_restored",
                {
                    "checkpoint_id": checkpoint_id,
                    "parent_rollout_id": parent_rollout_id,
                    "rollout_id": rollout_id,
                    "gold_rollout_id": gold_rollout_id,
                },
                caused_by=tuple(item for item in (rollout_started_event,) if item),
                structural={"checkpoint_id": checkpoint_id},
                actor="environment",
            )
            if restored_event:
                trace_refs["checkpoints"].append(restored_event)

        turns: list[dict[str, Any]] = []
        action_history: list[str] = []
        scheduled_checkpoints: list[dict[str, Any]] = []
        invalid_actions = 0
        truncated_calls = 0
        step = 0
        llm_calls = 0
        readout = await GOLD.readout(gold_rollout_id)
        observation_event = trace.event(
            "environment.observation",
            {
                "step": step,
                "observation_text": readout.get("observation_text"),
                "valid_actions": readout.get("valid_actions"),
                "public": readout.get("public"),
                "state": _state_from_readout(readout),
            },
            caused_by=tuple(item for item in (rollout_started_event,) if item),
            structural={"step": step},
            actor="environment",
        )
        if observation_event:
            trace_refs["observations"].append(observation_event)
        previous_achievements = set(_achievement_names(readout))

        while step < max_steps and llm_calls < max_llm_turns:
            state = _state_from_readout(readout)
            if state["terminated"] or state["truncated"]:
                break
            turn = await agent.choose_action(
                readout=readout,
                objective=objective,
                action_history=action_history,
                steps_remaining=max_steps - step,
                llm_calls_remaining=max_llm_turns - llm_calls,
                llm_call=llm_calls + 1,
            )
            llm_calls += 1
            if turn.proposal_event_id:
                trace_refs["proposals"].append(turn.proposal_event_id)
            if turn.invalid_parse or turn.repaired:
                invalid_actions += 1
            if turn.truncated:
                truncated_calls += 1
            planned_actions = [
                str(action) for action in (turn.actions or [turn.action]) if str(action)
            ]
            if not planned_actions:
                planned_actions = ["noop"]
            batch_size = min(len(planned_actions), max_steps - step)
            if batch_size < len(planned_actions):
                # A second place actions disappear: the step budget, after the
                # parser already applied its own cap. Both are decisions about
                # the model's plan and both are now on the record.
                trace.event(
                    "agent.actions_dropped",
                    {
                        "llm_call": llm_calls,
                        "declared_count": len(planned_actions),
                        "accepted_count": batch_size,
                        "dropped_actions": planned_actions[batch_size:],
                        "reason": "steps_remaining",
                        "steps_remaining": max_steps - step,
                    },
                    caused_by=tuple(
                        item for item in (turn.proposal_event_id,) if item
                    ),
                    structural={"llm_call": llm_calls},
                )
            for batch_index, action in enumerate(planned_actions[:batch_size]):
                action_event = trace.event(
                    "environment.action_executed",
                    {
                        "action": action,
                        "step": step,
                        "llm_call": llm_calls,
                        "batch_index": batch_index,
                        "batch_size": batch_size,
                    },
                    caused_by=tuple(
                        item
                        for item in (turn.proposal_event_id, observation_event)
                        if item
                    ),
                    structural={
                        "step": step,
                        "llm_call": llm_calls,
                        "batch_index": batch_index,
                    },
                    actor="environment",
                )
                if action_event:
                    trace_refs["executed_actions"].append(action_event)
                step_payload = await GOLD.step(gold_rollout_id, action)
                readout = step_payload.get("readout")
                if not isinstance(readout, dict):
                    readout = await GOLD.readout(gold_rollout_id)
                action_history.append(action)
                private = _private(readout)
                observation_event = trace.event(
                    "environment.observation",
                    {
                        "step": step + 1,
                        "observation_text": readout.get("observation_text"),
                        "valid_actions": readout.get("valid_actions"),
                        "public": readout.get("public"),
                        "state": _state_from_readout(readout),
                    },
                    caused_by=tuple(item for item in (action_event,) if item),
                    structural={"step": step + 1},
                    actor="environment",
                )
                if observation_event:
                    trace_refs["observations"].append(observation_event)
                # The engine's own account of what this action did, carried
                # through verbatim. Without it a reader can only diff
                # observations and call everything that did not visibly change
                # a "no-op", which is how 45 of 100 actions were classified with
                # no way to tell a refused craft from a blocked move.
                action_outcome = step_payload.get("outcome")
                action_outcome = action_outcome if isinstance(action_outcome, dict) else {}
                transition_event = trace.event(
                    "environment.transition",
                    {
                        "step": step,
                        "action": action,
                        "terminated": bool(private.get("terminated")),
                        "truncated": bool(private.get("truncated")),
                        "done_reason": private.get("done_reason"),
                        "total_reward": float(private.get("total_reward") or 0.0),
                        "effect": action_outcome.get("effect"),
                        "effect_reason": action_outcome.get("reason"),
                        "changed_state": action_outcome.get("changed"),
                        "effect_detail": action_outcome.get("detail"),
                        "resource_deltas": action_outcome.get("resource_deltas") or [],
                        "achievements_unlocked": action_outcome.get("achievements_unlocked") or [],
                        "rule_violations": action_outcome.get("rule_violations") or [],
                    },
                    caused_by=tuple(item for item in (action_event, observation_event) if item),
                    structural={"step": step},
                    actor="environment",
                )
                if transition_event:
                    trace_refs["transitions"].append(transition_event)
                current_achievements = set(_achievement_names(readout))
                new_achievements = sorted(current_achievements - previous_achievements)
                trace_refs["rewards"].extend(
                    _emit_achievement_rewards(
                        trace,
                        step=step,
                        new_achievements=new_achievements,
                        current_achievements=current_achievements,
                        total_reward=float(private.get("total_reward") or 0.0),
                        action_event=action_event,
                        observation_event=observation_event,
                    )
                )
                previous_achievements = current_achievements
                turns.append(
                    {
                        "ply": step,
                        "llm_call": llm_calls,
                        "action": action,
                        "assistant_text": turn.assistant_text
                        if batch_index == 0
                        else "",
                        "invalid_parse": turn.invalid_parse,
                        "repaired": turn.repaired,
                        "finish_reason": turn.finish_reason
                        if batch_index == 0
                        else None,
                        "truncated": turn.truncated,
                        "usage": turn.usage if batch_index == 0 else {},
                        "request_id": turn.request_id if batch_index == 0 else None,
                        "model": turn.model if batch_index == 0 else None,
                        "batch_size": batch_size,
                        "batch_index": batch_index,
                        "reward_total": float(private.get("total_reward") or 0.0),
                        "achievements": _achievement_names(readout),
                        "effect": action_outcome.get("effect"),
                        "effect_reason": action_outcome.get("reason"),
                        "changed_state": action_outcome.get("changed"),
                        # Declared vs executed, on the record that summarises the
                        # turn. Seed 202 declared eleven actions and ran ten.
                        "declared_actions": list(turn.declared_actions)
                        if batch_index == 0
                        else [],
                        "dropped_actions": list(turn.dropped_actions)
                        if batch_index == 0
                        else [],
                        "rejected_actions": list(turn.rejected_actions)
                        if batch_index == 0
                        else [],
                        "truncation_reason": turn.truncation_reason
                        if batch_index == 0
                        else None,
                    }
                )
                if progress_rollout_id:
                    _update_rollout_progress(
                        progress_rollout_id,
                        llm_calls=llm_calls,
                        max_llm_turns=max_llm_turns,
                        step=step,
                        max_steps=max_steps,
                        readout=readout,
                        turns=turns,
                        seed=seed,
                        metadata=request_metadata,
                    )
                step += 1
                private = _private(readout)
                if bool(private.get("terminated")) or bool(private.get("truncated")):
                    break
            if schedule_per_turn:
                cp_blob = await GOLD.checkpoint_with_blob(gold_rollout_id)
                cp_id = f"{checkpoint_prefix}_{llm_calls:04d}"
                checkpoint = _checkpoint_from_gold(
                    cp_id, rollout_id, cp_blob, task, llm_calls, readout
                )
                CHECKPOINTS[cp_id] = checkpoint
                scheduled_checkpoints.append(_public_checkpoint(checkpoint))
                checkpoint_event = trace.event(
                    "environment.checkpoint_created",
                    _public_checkpoint(checkpoint),
                    caused_by=tuple(item for item in (observation_event,) if item),
                    structural={"checkpoint_id": cp_id, "llm_call": llm_calls},
                    actor="environment",
                )
                if checkpoint_event:
                    trace_refs["checkpoints"].append(checkpoint_event)
            if bool(_private(readout).get("terminated")) or bool(
                _private(readout).get("truncated")
            ):
                break

        final_state = _state_from_readout(readout)
        labels = _objective_labels(readout, task)
        reward = float(_private(readout).get("total_reward") or 0.0)
        private = _private(readout)
        usage_totals = _turn_usage_totals(turns)
        # Four different things were one word. `terminated` is true when the
        # engine ended the episode — which includes the agent *dying* — so a
        # rollout that died at step 12 reported outcome "success", and every
        # budget-truncated run sat beside `success_status: success` too. The
        # rig completing is not the task succeeding, and neither is the step
        # budget running out.
        done_reason = str(private.get("done_reason") or "")
        outcome = (
            "boss_defeated"
            if done_reason == "boss_defeated"
            else "died"
            if done_reason == "death"
            else "budget_exhausted"
            if final_state["truncated"] or done_reason == "max_steps"
            else "terminated"
            if final_state["terminated"]
            else "unfinished"
        )
        task_success = (
            True
            if outcome == "boss_defeated"
            else False
            if outcome == "died"
            # A budget-truncated episode neither achieved the task nor failed at
            # it; it ran out of room. Calling that False would be as wrong as
            # calling it True.
            else None
        )
        terminal_checkpoint_id = f"{rollout_id}_terminal"
        terminal_blob = await GOLD.checkpoint_with_blob(gold_rollout_id)
        terminal_checkpoint = _checkpoint_from_gold(
            terminal_checkpoint_id,
            rollout_id,
            terminal_blob,
            task,
            llm_calls,
            readout,
        )
        CHECKPOINTS[terminal_checkpoint_id] = terminal_checkpoint
        terminal_checkpoint_event = trace.event(
            "environment.checkpoint_created",
            _public_checkpoint(terminal_checkpoint),
            caused_by=tuple(item for item in (observation_event,) if item),
            structural={"checkpoint_id": terminal_checkpoint_id, "terminal": True},
            actor="environment",
        )
        if terminal_checkpoint_event:
            trace_refs["checkpoints"].append(terminal_checkpoint_event)
        finished_event = trace.event(
            "environment.rollout_finished",
            {
                "rollout_id": rollout_id,
                "outcome": outcome,
                "reward": reward,
                "steps": step,
                "llm_calls": llm_calls,
                "achievements": _achievement_names(readout),
                "terminal_checkpoint_id": terminal_checkpoint_id,
            },
            caused_by=tuple(item for item in (observation_event,) if item),
            structural={"rollout_id": rollout_id},
            actor="environment",
        )
        if finished_event:
            trace_refs["lifecycle"].append(finished_event)
        return {
            "schema_version": "goex_rollout_response.v1",
            "rollout_id": rollout_id,
            "trace_correlation_id": trace_correlation_id,
            "trial_id": request.get("trial_id") or f"craftax-goex-{seed}",
            "status": "completed",
            # Infrastructure, not task: every sibling container uses this field
            # to mean "the rollout ran to a terminal state without rig failure".
            # Task success is `task_success`, which is separate on purpose.
            "success_status": "success",
            "status_detail": outcome,
            "task_success": task_success,
            "episode_end": outcome,
            "termination_reason": private.get("done_reason"),
            "parent_rollout_id": parent_rollout_id,
            "parent_checkpoint_id": checkpoint_id,
            "terminal_checkpoint_id": terminal_checkpoint_id,
            "reward_info": {
                "outcome_reward": reward,
                "details": {
                    "seed": seed,
                    "task_id": task.get("task_id"),
                    "objective": objective,
                    "outcome": outcome,
                    "steps": step,
                    "input_tokens": usage_totals["input_tokens"],
                    "prompt_tokens": usage_totals["prompt_tokens"],
                    "completion_tokens": usage_totals["completion_tokens"],
                    "llm_calls": llm_calls,
                    "call_accounting": usage_totals,
                    "invalid_action_count": invalid_actions,
                    # Distinguishes a budget failure from a formatting failure; a
                    # nonzero count means invalid_action_count is not about prompting.
                    "truncated_call_count": truncated_calls,
                    "model": agent.config.model,
                    "gold_rollout_id": gold_rollout_id,
                    "achievements": _achievement_names(readout),
                    "done_reason": private.get("done_reason"),
                    "task_success": task_success,
                    "episode_end": outcome,
                },
            },
            "summary": {
                "outcome": outcome,
                "task_success": task_success,
                "reward": reward,
                "steps": step,
                "max_steps": max_steps,
                "achievements": _achievement_names(readout),
            },
            "metadata": {
                **(
                    request.get("metadata")
                    if isinstance(request.get("metadata"), dict)
                    else {}
                ),
                "objective_labels": labels,
                "final_objective_labels": labels,
                "policy_llm_turns": llm_calls,
                "achievement_count": len(_achievement_names(readout)),
                "world": _world_identity(task),
            },
            "world": _world_identity(task),
            "objective_labels": labels,
            "final_objective_labels": labels,
            "state": final_state,
            "checkpoint": _public_checkpoint(terminal_checkpoint),
            "scheduled_checkpoints": scheduled_checkpoints,
            "trace_refs": trace_refs,
            "artifact": [{"artifact_type": "turns", "turns": turns}],
            "artifacts": [{"artifact_type": "turns", "turns": turns}],
        }
    except HTTPException:
        raise
    except Exception as exc:
        trace.event(
            "environment.rollout_failed",
            {
                "rollout_id": rollout_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            structural={"rollout_id": rollout_id},
            actor="environment",
        )
        _raise_gold_error(exc)
        raise
    finally:
        trace.close()


def _checkpoint_from_gold(
    checkpoint_id: str,
    rollout_id: str,
    gold_payload: dict[str, Any],
    task: dict[str, Any],
    policy_index: int,
    readout: dict[str, Any],
) -> dict[str, Any]:
    blob = str(gold_payload.get("blob") or "").strip()
    if not blob:
        raise RuntimeError(f"craftax checkpoint {checkpoint_id} missing blob")
    labels = _objective_labels(readout, task)
    achievements = _achievement_names(readout)
    return {
        "checkpoint_id": checkpoint_id,
        "rollout_id": rollout_id,
        "policy_llm_call_index": policy_index,
        "reward": float(_private(readout).get("total_reward") or 0.0),
        "achievements": achievements,
        "objective_labels": labels,
        "metadata": {
            "rollout_id": rollout_id,
            "policy_llm_call_index": policy_index,
            "achievements": achievements,
            "objective_labels": labels,
        },
        "blob_b64": blob,
    }


def _public_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in checkpoint.items() if key != "blob_b64"}


def _requested_rollout_id(request: Mapping[str, Any]) -> str | None:
    for key in ("rollout_id", "target_rollout_id", "trace_correlation_id"):
        value = str(request.get(key) or "").strip()
        if value and not value.startswith("$"):
            return value
    return None


def main() -> None:
    for path in (TASK_DIR, TASK_DIR / "gold_python", TASK_DIR / "shared"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8104"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
