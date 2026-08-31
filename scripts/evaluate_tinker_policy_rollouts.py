#!/usr/bin/env python3
"""Run multi-step Craftax policy rollouts with Tinker models."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from transformers import AutoTokenizer

from evaluate_tinker_action_puzzles import parse_json_object
from evaluate_tinker_native_observation import _decode_tokens, _model_input_from_tokens
from evaluate_tinker_state_puzzles import (
    ACTION_NAMES,
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

POLICY_SYSTEM_PROMPT = (
    "You are playing Craftax. Choose a sequence of 5 to 8 valid actions that maximizes unlocked "
    "achievements. The game will execute the actions in order before asking you again. "
    "Return only a JSON object with an actions list; do not explain your choice."
)
ACTION_PREFIX = '{"actions":["'
ACTION_ALIASES = {
    "move_left": "left",
    "move_right": "right",
    "move_up": "up",
    "move_down": "down",
    "interact": "do",
}

STRICT_OBSERVATION_FORMAT = """Observation constraints:
<observation>
scenario: one short sentence about the visible situation.
visible: comma-separated visible important objects with counts, like tree=5, stone=2. Include only things actually visible in the grid.
opportunities: comma-separated visible achievement ids with counts, like collect_wood=5. Include only ids supported by visible glyphs.
threats: comma-separated visible threats, or none.
</observation>
Do not mention absent objects, buildings, tools, plants, chests, ladders, monsters, or achievement ids.
An opportunity id may appear only when supported by a visible glyph."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_tinker() -> Any:
    import tinker as tinker_module

    return tinker_module


def load_tokenizer(model: str) -> Any:
    """Load the teacher tokenizer without allocating a trainable Tinker model."""
    return AutoTokenizer.from_pretrained(model, fast=True)


def achievement_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count) for key, count in value.items() if int(count) > 0}
    if isinstance(value, list):
        return {str(name): 1 for name in value}
    return {}


def achievement_deltas(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    deltas = {}
    for key, count in after.items():
        delta = int(count) - int(before.get(key, 0))
        if delta > 0:
            deltas[key] = delta
    return dict(sorted(deltas.items()))


def inventory_text(readout: dict[str, Any]) -> str:
    inventory = ((readout.get("observation") or {}).get("inventory") or {})
    parts = [
        f"{key}={value}"
        for key, value in inventory.items()
        if key not in {"potions", "learned_spells"} and value not in (0, {}, [], None)
    ]
    return ", ".join(parts) if parts else "empty"


def policy_prompt(readout: dict[str, Any], rows: list[str], view_size: int, call_index: int, max_calls: int) -> str:
    observation = readout["observation"]
    player = observation["player"]
    valid_actions = [str(action) for action in readout.get("valid_actions") or ACTION_NAMES]
    achieved = achievement_counts(observation.get("achievements"))
    visible_counts = visible_term_counts(rows)
    opportunities = visible_achievement_hint_counts(rows)
    return "\n".join(
        [
            f"Craftax policy rollout. decision={call_index + 1}/{max_calls}. view={view_size}x{view_size}.",
            "Goal: maximize newly unlocked achievements over the remaining decisions.",
            f"player: pos={player['pos']} direction={player['direction']} front_tile={player['front_tile']}",
            f"inventory: {inventory_text(readout)}",
            "achievements_so_far: " + compact_count_text(achieved),
            "legend: P player, . grass/path, , sand/gravel, ~ water, o stone, T tree, c coal, i iron, d diamond, C cow, S skeleton, Z zombie, a table, F furnace, L lava, p plant, R ripe plant, > down ladder, < up ladder, h chest, t torch, u fountain, # wall",
            "local_map:",
            *rows,
            "visible_counts: " + compact_count_text(visible_counts),
            "visible_achievement_opportunities: " + compact_count_text(opportunities),
            "valid_action_tokens: " + ", ".join(valid_actions),
            'Reply only with a JSON action list containing 5 to 8 valid actions, for example: '
            '{"actions":["do","right","right","right","right"]}',
        ]
    )


def extract_actions(text: str, valid_actions: list[str]) -> tuple[list[str], bool, dict[str, Any]]:
    valid_action_set = set(valid_actions)
    direct = re.search(r'"actions"\s*:\s*\[([^\]]*)\]', text)
    if direct is not None:
        actions = [
            ACTION_ALIASES.get(action.strip(), action.strip())
            for action in re.findall(r'"([^"]+)"', direct.group(1))
        ]
        if 5 <= len(actions) <= 8 and all(action in valid_action_set for action in actions):
            return actions, True, {"actions": actions}
    parsed, parse_ok = parse_json_object(text)
    raw_actions = parsed.get("actions")
    if isinstance(raw_actions, list):
        actions = [ACTION_ALIASES.get(str(action).strip(), str(action).strip()) for action in raw_actions]
        if 5 <= len(actions) <= 8 and all(action in valid_action_set for action in actions):
            return actions, parse_ok, parsed
    return ["noop"], False, parsed


def render_prompt(tokenizer: Any, prompt: str, *, model: str) -> list[int]:
    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if "gpt-oss" in model:
        rendered += "<|channel|>final<|message|>"
    return [
        int(token_id)
        for token_id in tokenizer(rendered + ACTION_PREFIX, add_special_tokens=False)["input_ids"]
    ]


async def sample_policy_text(
    sample_client: Any,
    tokenizer: Any,
    tinker_module: Any,
    prompt: str,
    *,
    model: str,
    max_tokens: int,
) -> tuple[str, str | None]:
    model_input = _model_input_from_tokens(
        tinker_module.ModelInput,
        render_prompt(tokenizer, prompt, model=model),
    )
    response = await sample_client.sample_async(
        model_input,
        num_samples=1,
        sampling_params=tinker_module.SamplingParams(
            # Sample at the budget the caller asked for. The former min(max_tokens, 96)
            # clamp meant the report's max_tokens field named a budget no call ever ran
            # at, and every call over 96 tokens came back truncated and unparseable.
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|return|>", "<|end|>"],
        ),
    )
    sequence = response.sequences[0]
    return ACTION_PREFIX + _decode_tokens(tokenizer, list(sequence.tokens)), getattr(
        sequence, "stop_reason", None
    )


async def run_rollout(
    game_client: httpx.AsyncClient,
    base_url: str,
    suite: dict[str, Any],
    sample_client: Any,
    tokenizer: Any,
    tinker_module: Any,
    *,
    model: str,
    seed: int,
    view_size: int,
    max_calls: int,
    max_tokens: int,
) -> dict[str, Any]:
    task = task_with_view_radius(suite, seed, view_size // 2)
    root = await http_json(game_client, base_url, "POST", "/rollouts", {"task": task, "seed": seed})
    rollout_id = str(root["rollout_id"])
    readout = root["readout"]
    before = achievement_counts(readout["observation"].get("achievements"))
    turns = []
    latest = root
    started = time.perf_counter()
    try:
        for call_index in range(max_calls):
            if latest.get("terminated") or latest.get("truncated"):
                break
            rows = crop_map([str(row) for row in readout["observation"].get("local_map") or []], view_size)
            valid_actions = [str(action) for action in readout.get("valid_actions") or ACTION_NAMES]
            prompt = policy_prompt(readout, rows, view_size, call_index, max_calls)
            call_started = time.perf_counter()
            assistant_text, stop_reason = await sample_policy_text(
                sample_client,
                tokenizer,
                tinker_module,
                prompt,
                model=model,
                max_tokens=max_tokens,
            )
            latency_ms = round((time.perf_counter() - call_started) * 1000, 2)
            planned_actions, parse_ok, parsed = extract_actions(assistant_text, valid_actions)
            executed_actions = []
            for action in planned_actions:
                latest = await http_json(
                    game_client, base_url, "POST", f"/rollouts/{rollout_id}/step", {"action": action}
                )
                executed_actions.append(action)
                readout = latest["readout"]
                if latest.get("terminated") or latest.get("truncated"):
                    break
            private = readout["private"]
            turns.append(
                {
                    "call_index": call_index,
                    "actions": planned_actions,
                    "executed_actions": executed_actions,
                    "steps_executed": len(executed_actions),
                    "parse_ok": parse_ok,
                    "parsed": parsed,
                    "valid_actions": valid_actions,
                    "assistant_text": assistant_text,
                    "stop_reason": stop_reason,
                    "call_truncated": str(stop_reason) == "length",
                    "latency_ms": latency_ms,
                    "reward_last": private.get("reward_last", 0.0),
                    "reward_total": private.get("total_reward", 0.0),
                    "achievements": achievement_counts(readout["observation"].get("achievements")),
                    "invalid_action_count": int(private.get("invalid_action_count", 0)),
                    "grid": rows,
                }
            )
        final_private = readout["private"]
        after = achievement_counts(readout["observation"].get("achievements"))
        deltas = achievement_deltas(before, after)
        return {
            "model": model,
            "seed": seed,
            "rollout_id": rollout_id,
            "view_size": view_size,
            "decision_calls": len(turns),
            "actions": [action for turn in turns for action in turn["executed_actions"]],
            "parse_rate": round(sum(1 for turn in turns if turn["parse_ok"]) / len(turns), 4) if turns else 0.0,
            # A call cut off at the token budget cannot emit an action batch, so it lands
            # in the parse_rate miss column. Report it separately or a budget failure
            # reads as a prompt failure.
            "call_truncation_rate": round(
                sum(1 for turn in turns if turn["call_truncated"]) / len(turns), 4
            )
            if turns
            else 0.0,
            "invalid_action_count": int(final_private.get("invalid_action_count", 0)),
            "reward": float(final_private.get("total_reward", 0.0)),
            "achievement_count": len(after),
            "achievement_delta": deltas,
            "achievements": after,
            "terminated": bool(latest.get("terminated", False)),
            "truncated": bool(latest.get("truncated", False)),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "turns": turns,
        }
    finally:
        try:
            await http_json(game_client, base_url, "DELETE", f"/rollouts/{rollout_id}", None)
        except httpx.HTTPError:
            pass


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(row["reward"]) for row in rows]
    calls = [int(row["decision_calls"]) for row in rows]
    return {
        "model": model,
        "n": len(rows),
        "reward_mean": round(statistics.mean(rewards), 6) if rewards else 0.0,
        "reward_sum": round(sum(rewards), 6),
        "achievement_count_mean": round(statistics.mean([int(row["achievement_count"]) for row in rows]), 4)
        if rows
        else 0.0,
        "decision_calls_mean": round(statistics.mean(calls), 4) if calls else 0.0,
        "parse_rate_mean": round(statistics.mean([float(row["parse_rate"]) for row in rows]), 4) if rows else 0.0,
        "call_truncation_rate_mean": round(
            statistics.mean([float(row["call_truncation_rate"]) for row in rows]), 4
        )
        if rows
        else 0.0,
        # The raw vocabulary, so a reader can check the truncation rate against what the
        # sampler actually returned instead of trusting one hardcoded string.
        "stop_reason_counts": {
            reason: sum(
                1
                for row in rows
                for turn in row["turns"]
                if str(turn["stop_reason"]) == reason
            )
            for reason in sorted(
                {str(turn["stop_reason"]) for row in rows for turn in row["turns"]}
            )
        },
        "invalid_action_count": sum(int(row["invalid_action_count"]) for row in rows),
        "achievement_delta_counts": {
            key: sum(int(row["achievement_delta"].get(key, 0)) for row in rows)
            for key in sorted({key for row in rows for key in row["achievement_delta"]})
        },
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise SystemExit("TINKER_API_KEY is required")
    validate_view_sizes([args.view_size])
    tinker_module = _load_tinker()
    service_client = tinker_module.ServiceClient(user_metadata={"task": "craftax_tinker_policy_rollouts"})
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    suite = json.loads(Path(args.suite).read_text())
    proc: subprocess.Popen[Any] | None = None
    base_url = args.base_url
    started = time.perf_counter()
    try:
        if not base_url:
            proc = spawn_rust_service(args.port)
            base_url = f"http://127.0.0.1:{args.port}"
        rows = []
        async with httpx.AsyncClient(limits=client_limits(), timeout=120.0) as game_client:
            for model in models:
                tokenizer = load_tokenizer(args.base_model if model.startswith("tinker://") else model)
                if model.startswith("tinker://"):
                    sample_client = await service_client.create_sampling_client_async(
                        model_path=model
                    )
                else:
                    sample_client = await service_client.create_sampling_client_async(
                        base_model=model
                    )
                for seed in seeds:
                    log(f"rollout model={model} seed={seed}")
                    rows.append(
                        await run_rollout(
                            game_client,
                            base_url,
                            suite,
                            sample_client,
                            tokenizer,
                            tinker_module,
                            model=model,
                            seed=seed,
                            view_size=args.view_size,
                            max_calls=args.max_calls,
                            max_tokens=args.max_tokens,
                        )
                    )
        report = {
            "schema": "gamebench.craftax.tinker_policy_rollouts.v1",
            "engine_lane": "rust",
            "models": models,
            "seeds": seeds,
            "view_size": args.view_size,
            "max_calls": args.max_calls,
            "max_tokens": args.max_tokens,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "summaries": [summarize_model(model, [row for row in rows if row["model"] == model]) for model in models],
            "rollouts": rows,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True)
    parser.add_argument("--base-model", default="openai/gpt-oss-20b")
    parser.add_argument("--seeds", default="501,502,503,504,505,506,507,508,509,510")
    parser.add_argument("--suite", default=str(TASK_DIR / "defaults" / "policy_sweep" / "policy_dev_v1.json"))
    parser.add_argument("--view-size", type=int, default=13)
    parser.add_argument("--max-calls", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=19117)
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--output",
        default=str(TASK_DIR / "reports" / "policy_sweep" / "tinker_policy_rollouts.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    print(
        json.dumps(
            {
                "elapsed_s": report["elapsed_s"],
                "models": report["models"],
                "seeds": report["seeds"],
                "view_size": report["view_size"],
                "max_calls": report["max_calls"],
                "summaries": report["summaries"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
