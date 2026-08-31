"""GEPA-facing orchestrator. Apply policy_script; Monty load; Rust episode."""

from __future__ import annotations

import argparse
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

from containers.gepa_codepolicy.apply import apply_source, sha256_text
from containers.gepa_codepolicy.rust_env import ENV_PROTOCOL

GEPA_OPTIMIZER_CONTRACT_VERSION = "synth_optimizers.gepa.v2"
TASK_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = TASK_ROOT / "policies" / "heuristic_baseline.py"
TRAIN_SEEDS = (101, 102, 103, 104, 105)
HELDOUT_SEEDS = (201, 202)
MAX_STEPS = 60
TASK_TEMPLATE = "tasks/policy_dev_template.json"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _task_id(payload: dict[str, Any]) -> tuple[str, str, int]:
    raw = str(payload.get("task_id") or payload.get("task", {}).get("task_id") or "train:0")
    split, _, seed_s = raw.partition(":")
    if split not in {"train", "heldout"}:
        split = str(payload.get("split") or "train")
    index = int(seed_s) if seed_s.isdigit() else 0
    pool = TRAIN_SEEDS if split == "train" else HELDOUT_SEEDS
    seed = pool[index % len(pool)]
    return f"{split}:{index}", split, seed


def _bundle_value(payload: dict[str, Any], field: str) -> Any:
    bundle = payload.get("lever_bundle") if isinstance(payload.get("lever_bundle"), dict) else {}
    values = bundle.get("values") if isinstance(bundle, dict) else None
    if isinstance(values, dict) and field in values:
        return values[field]
    candidate = payload.get("candidate")
    if isinstance(candidate, dict) and field in candidate:
        return candidate[field]
    return None


def _program(seed_source: str) -> dict[str, Any]:
    return {
        "version": "prompt_program.v1",
        "program_id": "craftax_monty_rust_code_policy",
        "modules": [
            {
                "module_id": "policy_script",
                "role": "other",
                "content": seed_source,
                "mutable": True,
                "candidate_field": "policy_script",
                "template_variables": [],
                "metadata": {
                    "lever_kind": "policy_script",
                    "protocol_id": "unified_diff.v1",
                    "constraints": {
                        "runtime": "python_source",
                        "load": "monty_python",
                        "entrypoint": "choose_actions",
                        "signature": ENV_PROTOCOL,
                        "path": "policies/candidate.py",
                        "module": "candidate",
                        "isolation": "isolated_policy_process",
                    },
                },
            }
        ],
        "target_modules": [
            {"module_id": "policy_script", "candidate_field": "policy_script", "objective": "outcome_reward"}
        ],
        "seed_candidate": {"policy_script": seed_source},
        "rollout_overlay_schema": {"candidate_fields": ["policy_script"]},
        "side_info_schemas": [
            {"schema_id": "code_policy_game_trace.v1", "when": "terminal_rollout", "purpose": "proposer_actionable"},
            {"schema_id": "apply_report.v1", "when": "terminal_rollout", "purpose": "proposer_actionable"},
            {"schema_id": "monty_load.v1", "when": "terminal_rollout", "purpose": "proposer_actionable"},
        ],
        "metadata": {
            "env_id": "gamebench.craftax-singleplayer",
            "env_protocol": ENV_PROTOCOL,
            "substrate": "rust",
            "apply_isolation": "serial_reload",
        },
    }


def create_app(*, env_url: str, policy_url: str) -> FastAPI:
    app = FastAPI(title="craftax-gepa-monty-rust")
    env_url = env_url.rstrip("/")
    policy_url = policy_url.rstrip("/")
    seed_source = SEED_PATH.read_text(encoding="utf-8") if SEED_PATH.is_file() else ""
    current_source = seed_source
    records: dict[str, dict[str, Any]] = {}

    def _env_health() -> dict[str, Any]:
        return httpx.get(f"{env_url}/health", timeout=5.0).json()

    def _policy_health() -> dict[str, Any]:
        return httpx.get(f"{policy_url}/health", timeout=5.0).json()

    @app.get("/health")
    def health() -> dict[str, Any]:
        env = _env_health()
        policy = _policy_health()
        return {
            "status": "ok",
            "contract_version": GEPA_OPTIMIZER_CONTRACT_VERSION,
            "env": env,
            "policy": policy,
            "env_protocol": ENV_PROTOCOL,
        }

    @app.get("/metadata")
    @app.get("/info")
    def metadata() -> dict[str, Any]:
        body = _program(seed_source)
        return {
            "runtime": {
                "runtime_id": "craftax_monty_rust",
                "name": "Craftax Monty Python on gold_rust",
            },
            "capabilities": {
                "contract_version": "container_contract.v1",
                "rollout_modes": ["blocking", "sync"],
            },
            "metadata": {
                "optimizer_contracts": {
                    "gepa": {
                        "version": GEPA_OPTIMIZER_CONTRACT_VERSION,
                        "program_route": "/program",
                        "taskset_route": "/taskset",
                        "taskset_tasks_route": "/taskset/tasks",
                        "rollout_route": "/rollout",
                    }
                },
                "env_protocol": ENV_PROTOCOL,
                "side_info_schemas": body.get("side_info_schemas"),
                "apply_isolation": "serial_reload",
            },
        }

    @app.get("/program")
    def program() -> dict[str, Any]:
        return _program(seed_source)

    @app.get("/taskset")
    def taskset() -> dict[str, Any]:
        return {"taskset_id": "craftax_monty_rust_smoke", "splits": {"train": len(TRAIN_SEEDS), "heldout": len(HELDOUT_SEEDS)}}

    @app.post("/taskset/tasks")
    def taskset_tasks(payload: dict[str, Any]) -> dict[str, Any]:
        tasks = []
        for raw in payload.get("task_ids") or []:
            task_id, split, seed = _task_id({"task_id": str(raw)})
            tasks.append({"task_id": task_id, "split": split, "seed": seed})
        return {"tasks": tasks}

    @app.post("/rollout")
    def rollout(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal current_source
        task_id, split, seed = _task_id(payload)
        rollout_id = str(payload.get("rollout_id") or f"craftax_monty_{uuid.uuid4().hex[:12]}")
        side_info: list[dict[str, Any]] = []
        apply_failed = False
        env_before = _env_health()
        value = _bundle_value(payload, "policy_script")
        source, report, protocol = apply_source(current_source, value)
        report["lever_ids"] = ["policy_script"]
        if not report.get("patch_ok"):
            apply_failed = True
            episode: dict[str, Any] = {"reward": 0.0, "achievements": [], "turns": [], "events": []}
            side_info.append(report)
        else:
            loaded = httpx.post(
                f"{policy_url}/load",
                json={
                    "kind": "monty_python",
                    "module": "candidate",
                    "entry": "choose_actions",
                    "source": source,
                },
                timeout=60.0,
            )
            if loaded.status_code >= 400:
                apply_failed = True
                try:
                    detail = loaded.json()
                except Exception:  # noqa: BLE001
                    detail = {"error": loaded.text}
                if isinstance(detail.get("detail"), dict):
                    detail = detail["detail"]
                elif isinstance(detail.get("detail"), str):
                    detail = {"error": detail["detail"]}
                report["compile_ok"] = False
                report["patch_ok"] = True
                report["reject_reason"] = str(detail.get("error") or detail.get("detail") or "load_failed")
                episode = {"reward": 0.0, "achievements": [], "turns": [], "events": []}
                side_info.append(report)
            else:
                load_body = loaded.json()
                current_source = source
                report["compile_ok"] = True
                report["content_hash"] = load_body.get("content_hash")
                env_after = _env_health()
                report["env_untouched"] = env_before == env_after or (
                    env_before.get("lane") == env_after.get("lane") and env_after.get("ok") is True
                )
                report["env_before"] = env_before
                report["env_after"] = env_after
                side_info.append(report)
                side_info.append(
                    {
                        "schema_id": "monty_load.v1",
                        "lever_ids": ["policy_script"],
                        "summary": {
                            "kind": "monty_python",
                            "module": "candidate",
                            "entry": "choose_actions",
                            "sandbox": (load_body.get("isolation") or {}).get("sandbox"),
                        },
                        "body": {"spec": load_body.get("spec"), "isolation": load_body.get("isolation")},
                    }
                )
                episode = httpx.post(
                    f"{policy_url}/episode",
                    json={
                        "env_url": env_url,
                        "seed": seed,
                        "task_template": TASK_TEMPLATE,
                        "max_steps": int(payload.get("max_steps") or MAX_STEPS),
                    },
                    timeout=180.0,
                ).json()

        env_final = _env_health()
        if apply_failed:
            reward = 0.0
            success = "apply_failed"
            status = "failed"
        else:
            reward = float(episode.get("reward") or 0.0)
            success = "succeeded"
            status = "completed"
        achievements = list(episode.get("achievements") or [])
        side_info.append(
            {
                "schema_id": "code_policy_game_trace.v1",
                "lever_ids": ["policy_script"],
                "summary": {
                    "compile_ok": not apply_failed,
                    "ticks": episode.get("ticks"),
                    "achievements": achievements,
                    "stop_reason": episode.get("stop_reason"),
                    "architecture": episode.get("architecture"),
                },
                "body": {
                    "turns": (episode.get("turns") or [])[:40],
                    "events": episode.get("events") or [],
                },
            }
        )
        record = {
            "rollout_id": rollout_id,
            "status": status,
            "success_status": success,
            "task_id": task_id,
            "seed": seed,
            "split": split,
            "reward": reward,
            "reward_info": {
                "outcome_reward": reward,
                "metrics": {"achievements": achievements, "ticks": episode.get("ticks")},
            },
            "summary": {
                "outcome_reward": reward,
                "achievements": achievements,
                "protocol_id": protocol,
            },
            "trace": {"event_history": [{"type": "policy_act", **turn} for turn in (episode.get("turns") or [])]},
            "side_info": side_info,
            "actionable_side_info": side_info,
            "created_at": _now(),
            "env": env_final,
        }
        records[rollout_id] = record
        return record

    @app.get("/rollouts/{rollout_id}")
    def get_rollout(rollout_id: str) -> dict[str, Any]:
        if rollout_id not in records:
            raise HTTPException(status_code=404, detail="unknown rollout")
        return records[rollout_id]

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19300)
    parser.add_argument("--env-url", default=os.environ.get("GAMEBENCH_CRAFTAX_RUST_URL", "http://127.0.0.1:8098"))
    parser.add_argument("--policy-url", default=os.environ.get("CRAFTAX_MONTY_POLICY_URL", "http://127.0.0.1:19310"))
    args = parser.parse_args()
    app = create_app(env_url=args.env_url, policy_url=args.policy_url)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
