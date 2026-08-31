#!/usr/bin/env python3
"""Evaluate Tinker GPT-OSS models on Craftax action-sequence puzzles."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from evaluate_tinker_state_puzzles import (
    ACTION_NAMES,
    DEFAULT_MODELS,
    DEFAULT_URL,
    TASK_DIR,
    client_limits,
    compact_count_text,
    crop_map,
    http_json,
    spawn_rust_service,
    task_with_view_radius,
    validate_view_sizes,
    visible_achievement_hint_counts,
    visible_term_counts,
)


SYSTEM_PROMPT = (
    "You are solving a Craftax planning puzzle. Given a symbolic local_map, return "
    "one JSON object with key actions. actions must be a list of exact valid action "
    "tokens. The sequence should maximize total newly unlocked achievements when "
    "executed from the current state. Do not explain."
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


async def call_model(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    response = await client.post(
        DEFAULT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=120.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"{response.status_code} {response.reason_phrase}: {response.text[:1000]}",
            request=response.request,
            response=response,
        )
    payload = response.json()
    message = payload["choices"][0]["message"]
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "assistant_text": str(content),
        "usage": payload.get("usage", {}),
        "request_id": payload.get("id"),
        "latency_ms": latency_ms,
    }


def parse_json_object(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {"actions": value}), True
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        return {}, False
    try:
        value = json.loads(raw[start : end + 1])
        return (value if isinstance(value, dict) else {"actions": value}), True
    except json.JSONDecodeError:
        return {}, False


def extract_actions(parsed: dict[str, Any], max_actions: int) -> list[str]:
    raw = parsed.get("actions")
    if raw is None:
        raw = parsed.get("action_sequence")
    if raw is None:
        raw = parsed.get("sequence")
    if not isinstance(raw, list):
        return []
    actions = []
    valid = set(ACTION_NAMES)
    for item in raw:
        action = str(item).strip()
        if action in valid:
            actions.append(action)
        if len(actions) >= max_actions:
            break
    return actions


def achievement_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count) for key, count in value.items() if int(count) > 0}
    if isinstance(value, list):
        return {str(name): 1 for name in value}
    return {}


def log_count_score(before: dict[str, int], after: dict[str, int]) -> tuple[float, dict[str, int]]:
    deltas = {}
    for name, count in after.items():
        delta = int(count) - int(before.get(name, 0))
        if delta > 0:
            deltas[name] = delta
    return round(sum(math.log1p(delta) for delta in deltas.values()), 6), dict(sorted(deltas.items()))


def log_count_potential(counts: dict[str, int]) -> float:
    return round(sum(math.log1p(int(count)) for count in counts.values() if int(count) > 0), 6)


def action_prompt(readout: dict[str, Any], rows: list[str], view_size: int, max_actions: int) -> str:
    observation = dict(readout["observation"])
    inventory = observation.get("inventory") or {}
    inventory_text = ", ".join(
        f"{key}={value}"
        for key, value in inventory.items()
        if key not in {"potions", "learned_spells"} and value not in (0, {}, [], None)
    )
    if not inventory_text:
        inventory_text = "empty"
    visible_counts = visible_term_counts(rows)
    opportunity_counts = visible_achievement_hint_counts(rows)
    valid_actions = [str(action) for action in readout.get("valid_actions") or ACTION_NAMES]
    return "\n".join(
        [
            f"Craftax action puzzle ({view_size}x{view_size}).",
            f"goal: choose up to {max_actions} actions to maximize sum(log1p(new_count)) over newly unlocked achievement ids.",
            f"player: pos={observation['player']['pos']} direction={observation['player']['direction']} front_tile={observation['player']['front_tile']}",
            f"inventory: {inventory_text}",
            "legend: P player, . grass/path, , sand/gravel, ~ water, o stone, T tree, c coal, i iron, d diamond, C cow, S skeleton, Z zombie, a table, F furnace, L lava, p plant, R ripe plant, > down ladder, < up ladder, h chest, t torch, u fountain, # wall",
            "local_map:",
            *rows,
            "visible_counts: " + compact_count_text(visible_counts),
            "visible_achievement_opportunities: " + compact_count_text(opportunity_counts),
            "valid_action_tokens: " + ", ".join(valid_actions),
            'Reply as JSON only: {"actions":["do","right","do"]}.',
        ]
    )


async def sample_checkpoint(
    client: httpx.AsyncClient,
    base_url: str,
    suite: dict[str, Any],
    *,
    seed: int,
    max_steps_between: int,
    view_radius: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    task = task_with_view_radius(suite, seed, view_radius)
    payload = await http_json(client, base_url, "POST", "/rollouts", {"task": task, "seed": seed})
    rollout_id = str(payload["rollout_id"])
    try:
        readout = await http_json(client, base_url, "GET", f"/rollouts/{rollout_id}/readout")
        valid = [str(item) for item in readout.get("valid_actions") or ["noop"]]
        for _ in range(rng.randint(0, max_steps_between)):
            action = rng.choice(valid)
            stepped = await http_json(
                client,
                base_url,
                "POST",
                f"/rollouts/{rollout_id}/step",
                {"action": action},
            )
            stepped_readout = stepped.get("readout")
            if isinstance(stepped_readout, dict):
                readout = stepped_readout
            if stepped.get("terminated") or stepped.get("truncated"):
                break
            valid = [str(item) for item in (stepped.get("readout") or {}).get("valid_actions") or valid]
        readout = await http_json(
            client,
            base_url,
            "GET",
            f"/rollouts/{rollout_id}/readout",
        )
        checkpoint = await http_json(
            client,
            base_url,
            "POST",
            f"/rollouts/{rollout_id}/checkpoint",
            None,
        )
        return {
            "seed": seed,
            "rollout_id": rollout_id,
            "readout": readout,
            "checkpoint_blob": str(checkpoint["blob"]),
            "local_map_full": [str(row) for row in readout["observation"].get("local_map") or []],
            "achievement_counts_before": achievement_counts(readout["observation"].get("achievements")),
        }
    except Exception:
        await http_json(client, base_url, "DELETE", f"/rollouts/{rollout_id}", None)
        raise


async def evaluate_one(
    inference_client: httpx.AsyncClient,
    game_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    view_size: int,
    *,
    max_tokens: int,
    max_actions: int,
) -> dict[str, Any]:
    rows = crop_map(sample["local_map_full"], view_size)
    visible_opportunity_counts = visible_achievement_hint_counts(rows)
    potential_score = log_count_potential(visible_opportunity_counts)
    prompt = action_prompt(sample["readout"], rows, view_size, max_actions)
    assistant_text = ""
    parsed: dict[str, Any] = {}
    parse_ok = False
    actions: list[str] = []
    usage: dict[str, Any] = {}
    latency_ms = None
    error = None
    sim_result: dict[str, Any] = {}
    score = 0.0
    achievement_delta: dict[str, int] = {}
    try:
        inference = await call_model(inference_client, api_key, model, prompt, max_tokens=max_tokens)
        assistant_text = str(inference["assistant_text"])
        usage = dict(inference.get("usage") or {})
        latency_ms = float(inference["latency_ms"])
        parsed, parse_ok = parse_json_object(assistant_text)
        actions = extract_actions(parsed, max_actions)
        simulated = await http_json(
            game_client,
            base_url,
            "POST",
            f"/rollouts/{sample['rollout_id']}/simulate",
            {"blob": sample["checkpoint_blob"], "sequences": [actions]},
            timeout_s=120.0,
        )
        sim_result = dict((simulated.get("results") or [{}])[0])
        after = achievement_counts(sim_result.get("achievements"))
        score, achievement_delta = log_count_score(sample["achievement_counts_before"], after)
    except Exception as exc:
        error = str(exc)
    return {
        "model": model,
        "seed": sample["seed"],
        "view_size": view_size,
        "grid": rows,
        "prompt": prompt,
        "assistant_text": assistant_text,
        "parsed": parsed,
        "parse_ok": parse_ok,
        "actions": actions,
        "action_count": len(actions),
        "valid_action_rate": round(len(actions) / len(parsed.get("actions", []) or actions), 4)
        if parsed
        else 0.0,
        "score_log1p_new_achievements": score,
        "potential_log1p_visible_achievements": potential_score,
        "realized_over_potential": round(score / potential_score, 6) if potential_score > 0 else None,
        "visible_achievement_opportunity_counts": visible_opportunity_counts,
        "achievement_delta": achievement_delta,
        "sim_result": {
            "achievements": sim_result.get("achievements", []),
            "reward": sim_result.get("reward"),
            "reward_trace": sim_result.get("reward_trace", []),
            "terminated": sim_result.get("terminated"),
            "truncated": sim_result.get("truncated"),
            "steps": sim_result.get("steps"),
        },
        "usage": usage,
        "latency_ms": latency_ms,
        "error": error,
    }


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view = {}
    for view_size in sorted({int(row["view_size"]) for row in rows}):
        subset = [row for row in rows if int(row["view_size"]) == view_size]
        scores = [float(row["score_log1p_new_achievements"]) for row in subset]
        potentials = [float(row.get("potential_log1p_visible_achievements", 0.0) or 0.0) for row in subset]
        realized = sum(scores)
        potential = sum(potentials)
        by_view[str(view_size)] = {
            "n": len(subset),
            "parse_rate": round(sum(1 for row in subset if row["parse_ok"]) / len(subset), 4) if subset else 0.0,
            "score_mean": round(statistics.mean(scores), 6) if scores else 0.0,
            "potential_mean": round(statistics.mean(potentials), 6) if potentials else 0.0,
            "realized_over_potential": round(realized / potential, 6) if potential > 0 else None,
            "achievement_delta_counts": {
                key: sum(int(row["achievement_delta"].get(key, 0)) for row in subset)
                for key in sorted({key for row in subset for key in row["achievement_delta"]})
            },
            "latency_ms_mean": round(statistics.mean([row["latency_ms"] for row in subset if row["latency_ms"] is not None]), 2)
            if any(row["latency_ms"] is not None for row in subset)
            else None,
        }
    return {
        "model": model,
        "n": len(rows),
        "parse_rate": round(sum(1 for row in rows if row["parse_ok"]) / len(rows), 4) if rows else 0.0,
        "score_mean": round(statistics.mean([float(row["score_log1p_new_achievements"]) for row in rows]), 6)
        if rows
        else 0.0,
        "potential_mean": round(
            statistics.mean([float(row.get("potential_log1p_visible_achievements", 0.0) or 0.0) for row in rows]),
            6,
        )
        if rows
        else 0.0,
        "realized_over_potential": round(
            sum(float(row["score_log1p_new_achievements"]) for row in rows)
            / sum(float(row.get("potential_log1p_visible_achievements", 0.0) or 0.0) for row in rows),
            6,
        )
        if rows and sum(float(row.get("potential_log1p_visible_achievements", 0.0) or 0.0) for row in rows) > 0
        else None,
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "by_view_size": by_view,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key and not args.sample_only:
        raise SystemExit("TINKER_API_KEY is required")
    models = [part.strip() for part in args.models.split(",") if part.strip()] or DEFAULT_MODELS
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    view_sizes = validate_view_sizes(
        [int(part.strip()) for part in args.view_sizes.split(",") if part.strip()]
    )
    max_view = max(view_sizes)
    suite = json.loads(Path(args.suite).read_text())
    proc: subprocess.Popen[Any] | None = None
    base_url = args.base_url
    started = time.perf_counter()
    try:
        if not base_url:
            proc = spawn_rust_service(args.port)
            base_url = f"http://127.0.0.1:{args.port}"
        rows = []
        samples = []
        async with httpx.AsyncClient(limits=client_limits(), timeout=120.0) as game_client:
            for seed in seeds:
                log(f"sampling seed={seed}")
                sample = await sample_checkpoint(
                    game_client,
                    base_url,
                    suite,
                    seed=seed,
                    max_steps_between=args.max_steps_between,
                    view_radius=max_view // 2,
                )
                samples.append(sample)
            if not args.sample_only:
                async with httpx.AsyncClient(limits=client_limits(), timeout=120.0) as inference_client:
                    for model in models:
                        for sample in samples:
                            for view_size in view_sizes:
                                log(f"eval model={model} seed={sample['seed']} view={view_size}")
                                rows.append(
                                    await evaluate_one(
                                        inference_client,
                                        game_client,
                                        base_url,
                                        api_key,
                                        model,
                                        sample,
                                        view_size,
                                        max_tokens=args.max_tokens,
                                        max_actions=args.max_actions,
                                    )
                                )
            for sample in samples:
                try:
                    await http_json(game_client, base_url, "DELETE", f"/rollouts/{sample['rollout_id']}", None)
                except httpx.HTTPError:
                    pass
        report = {
            "schema": "gamebench.craftax.tinker_action_puzzle_eval.v1",
            "engine_lane": "rust",
            "inference_provider": "tinker",
            "models": models,
            "seeds": seeds,
            "view_sizes": view_sizes,
            "max_actions": args.max_actions,
            "max_tokens": args.max_tokens,
            "sample_only": bool(args.sample_only),
            "sample_count": len(samples),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "summaries": [summarize_model(model, [row for row in rows if row["model"] == model]) for model in models],
            "samples": [
                {
                    "seed": sample["seed"],
                    "achievement_counts_before": sample["achievement_counts_before"],
                    "local_map_full": sample["local_map_full"],
                }
                for sample in samples
            ],
            "results": rows,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--view-sizes", default="5,13,19")
    parser.add_argument("--suite", default=str(TASK_DIR / "defaults" / "policy_sweep" / "policy_dev_v1.json"))
    parser.add_argument("--max-steps-between", type=int, default=0)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--port", type=int, default=19114)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument(
        "--output",
        default=str(TASK_DIR / "reports" / "state_puzzles" / "tinker_gpt_oss_action_puzzles.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    print(
        json.dumps(
            {
                "elapsed_s": report["elapsed_s"],
                "models": report["models"],
                "view_sizes": report["view_sizes"],
                "sample_count": report["sample_count"],
                "summaries": report["summaries"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
