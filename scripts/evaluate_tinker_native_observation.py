#!/usr/bin/env python3
"""Evaluate Craftax observation prompts through Tinker's native sampler."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from evaluate_tinker_state_puzzles import (
    OBSERVATION_SYSTEM_PROMPT,
    TASK_DIR,
    client_limits,
    crop_map,
    current_achievement_counts,
    observation_prompt,
    sample_states,
    score_answer,
    spawn_rust_service,
    validate_view_sizes,
    visible_achievement_hint_counts,
    visible_achievement_hints,
    visible_term_counts,
    visible_terms,
)

STRICT_OBSERVATION_FORMAT = """Use this exact shape:
<observation>
scenario: one short sentence about the visible situation.
visible: comma-separated visible important objects with counts, like tree=5, stone=2. Include only things actually visible in the grid.
opportunities: comma-separated visible achievement ids with counts, like collect_wood=5. Include only ids supported by visible glyphs.
threats: comma-separated visible threats, or none.
</observation>
Do not mention absent objects, buildings, tools, plants, chests, ladders, monsters, or achievement ids. Do not choose actions."""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _load_tinker() -> Any:
    import tinker as tinker_module

    return tinker_module


def _model_input_from_tokens(model_input_cls: Any, tokens: list[int]) -> Any:
    try:
        return model_input_cls.from_ints(tokens=tokens)
    except TypeError:
        return model_input_cls.from_ints(tokens)


def _decode_tokens(tokenizer: Any, tokens: list[int]) -> str:
    text = tokenizer.decode(tokens)
    for marker in ("<|return|>", "<|end|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def native_sample(
    sample_client: Any,
    tokenizer: Any,
    tinker_module: Any,
    prompt: str,
    *,
    max_tokens: int,
) -> tuple[str, str | None]:
    messages = [
        {"role": "system", "content": OBSERVATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered = (
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        + "<|channel|>final<|message|>"
    )
    token_ids = [int(token_id) for token_id in tokenizer(rendered, add_special_tokens=False)["input_ids"]]
    model_input = _model_input_from_tokens(tinker_module.ModelInput, token_ids)
    response = sample_client.sample(
        model_input,
        num_samples=1,
        sampling_params=tinker_module.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=["<|return|>", "</observation>"],
        ),
    ).result()
    sequence = response.sequences[0]
    text = _decode_tokens(tokenizer, list(sequence.tokens))
    if "</observation>" not in text and "<observation>" in text:
        text = text + "\n</observation>"
    return text, getattr(sequence, "stop_reason", None)


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view = {}
    for view_size in sorted({int(row["view_size"]) for row in rows}):
        subset = [row for row in rows if int(row["view_size"]) == view_size]
        by_view[str(view_size)] = {
            "n": len(subset),
            "important_recall": mean_present(subset, "important_recall"),
            "important_count_exact_recall": mean_present(subset, "important_count_exact_recall"),
            "achievement_hint_recall": mean_present(subset, "achievement_hint_recall"),
            "achievement_count_exact_recall": mean_present(subset, "achievement_count_exact_recall"),
            "latency_ms_mean": round(statistics.mean([row["latency_ms"] for row in subset]), 2) if subset else None,
        }
    return {
        "model": model,
        "n": len(rows),
        "important_recall": mean_present(rows, "important_recall"),
        "important_count_exact_recall": mean_present(rows, "important_count_exact_recall"),
        "achievement_hint_recall": mean_present(rows, "achievement_hint_recall"),
        "achievement_count_exact_recall": mean_present(rows, "achievement_count_exact_recall"),
        "by_view_size": by_view,
    }


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["scores"][key] for row in rows if row["scores"].get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise SystemExit("TINKER_API_KEY is required")
    tinker_module = _load_tinker()
    service_client = tinker_module.ServiceClient(
        user_metadata={"task": "craftax_native_observation_eval"}
    )
    tokenizer_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=1,
        seed=7,
        train_mlp=False,
        train_attn=True,
        train_unembed=False,
    )
    tokenizer = tokenizer_client.get_tokenizer()
    models = [part.strip() for part in args.models.split(",") if part.strip()]
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    view_sizes = validate_view_sizes(
        [int(part.strip()) for part in args.view_sizes.split(",") if part.strip()]
    )
    suite = json.loads(Path(args.suite).read_text())
    proc: subprocess.Popen[Any] | None = None
    base_url = args.base_url
    started = time.perf_counter()
    try:
        if not base_url:
            proc = spawn_rust_service(args.port)
            base_url = f"http://127.0.0.1:{args.port}"
        samples = []
        async with httpx.AsyncClient(limits=client_limits(), timeout=60.0) as game_client:
            for seed in seeds:
                log(f"sampling seed={seed}")
                samples.extend(
                    await sample_states(
                        game_client,
                        base_url,
                        suite,
                        seed=seed,
                        states_per_seed=args.states_per_seed,
                        max_steps_between=args.max_steps_between,
                        view_radius=max(view_sizes) // 2,
                    )
                )
        rows = []
        for model in models:
            if model.startswith("tinker://"):
                sample_client = service_client.create_sampling_client(model_path=model)
            else:
                sample_client = service_client.create_sampling_client(base_model=model)
            for sample in samples:
                for view_size in view_sizes:
                    log(f"eval model={model} seed={sample['seed']} sample={sample['sample_index']} view={view_size}")
                    grid = crop_map(sample.get("local_map_full") or sample["local_map_13"], view_size)
                    labels = {
                        "important_terms": sorted(visible_terms(grid)),
                        "important_counts": visible_term_counts(grid),
                        "achievement_hints": sorted(visible_achievement_hints(grid)),
                        "achievement_hint_counts": visible_achievement_hint_counts(grid),
                        "current_achievement_counts": current_achievement_counts(sample["readout"]),
                        "immediate_actions": sample["immediate_actions"],
                        "immediate_achievements": sample["immediate_achievements"],
                    }
                    prompt = observation_prompt(sample["readout"], grid, view_size)
                    if args.prompt_format == "strict":
                        prompt = f"{prompt}\n\n{STRICT_OBSERVATION_FORMAT}"
                    start = time.perf_counter()
                    text, stop_reason = native_sample(
                        sample_client,
                        tokenizer,
                        tinker_module,
                        prompt,
                        max_tokens=args.max_tokens,
                    )
                    rows.append(
                        {
                            "model": model,
                            "seed": sample["seed"],
                            "sample_index": sample["sample_index"],
                            "view_size": view_size,
                            "grid": grid,
                            "assistant_text": text,
                            "stop_reason": stop_reason,
                            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                            "labels": labels,
                            "scores": score_answer({"observation": text}, labels),
                        }
                    )
        report = {
            "schema": "gamebench.craftax.tinker_native_observation_eval.v1",
            "models": models,
            "seeds": seeds,
            "view_sizes": view_sizes,
            "sample_count": len(samples),
            "max_tokens": args.max_tokens,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "summaries": [summarize_model(model, [row for row in rows if row["model"] == model]) for model in models],
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
    parser.add_argument("--models", required=True)
    parser.add_argument("--base-model", default="openai/gpt-oss-20b")
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--states-per-seed", type=int, default=2)
    parser.add_argument("--max-steps-between", type=int, default=12)
    parser.add_argument("--view-sizes", default="5,9,13,15,19")
    parser.add_argument("--suite", default=str(TASK_DIR / "defaults" / "policy_sweep" / "policy_dev_v1.json"))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--prompt-format", choices=("freeform", "strict"), default="freeform")
    parser.add_argument("--port", type=int, default=19116)
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--output",
        default=str(TASK_DIR / "reports" / "state_puzzles" / "tinker_native_observation_eval.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    print(
        json.dumps(
            {
                "elapsed_s": report["elapsed_s"],
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
