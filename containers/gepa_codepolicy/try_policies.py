#!/usr/bin/env python3
"""Author policy source locally, then POST it the way GEPA does (whole_file.v1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

from containers.gepa_codepolicy.apply import sha256_text
from containers.gepa_codepolicy.stack import start_stack

TASK_ROOT = Path(__file__).resolve().parents[2]
GAMEBENCH = TASK_ROOT.parents[1]
HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
CANDIDATES = HERE / "candidates"
BASELINE = TASK_ROOT / "policies" / "heuristic_baseline.py"
HARBOR = (
    GAMEBENCH
    / "adapters/harbor/bundles/code_policy_deo_hillclimb/solution/references/craftax-singleplayer/heuristic_policy.py"
)


def _gepa_whole_file(source: str, *, task_id: str = "train:0", max_steps: int = 80) -> dict:
    """Same envelope GEPA sends: lever_bundle.values.policy_script / whole_file.v1."""
    return {
        "task_id": task_id,
        "max_steps": max_steps,
        "lever_bundle": {
            "schema_version": "lever_bundle.v1",
            "values": {
                "policy_script": {
                    "protocol_id": "whole_file.v1",
                    "content": source,
                    "content_hash": sha256_text(source),
                }
            },
        },
    }


def _stitch_harbor(wrapper: str, baseline: str) -> str:
    """One HTTP payload: Harbor remap + baseline body. Isolated child cannot import."""
    renamed = baseline.replace("def choose_actions(", "def _reference_choose_actions(", 1)
    lines = wrapper.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("def choose_actions"))
    return renamed + "\n\n" + "\n".join(lines[start:]) + "\n"


def _summarize(name: str, body: dict) -> dict:
    trace = next((i for i in body.get("side_info") or [] if i.get("schema_id") == "code_policy_game_trace.v1"), {})
    apply_report = next((i for i in body.get("side_info") or [] if i.get("schema_id") == "apply_report.v1"), {})
    monty = next((i for i in body.get("side_info") or [] if i.get("schema_id") == "monty_load.v1"), {})
    turns = (trace.get("body") or {}).get("turns") or []
    return {
        "library": name,
        "status": body.get("success_status"),
        "reward": body.get("reward"),
        "achievements": (body.get("reward_info") or {}).get("metrics", {}).get("achievements"),
        "ticks": (trace.get("summary") or {}).get("ticks"),
        "actions": [row.get("action") for row in turns[:12]],
        "env_untouched": apply_report.get("env_untouched"),
        "protocol": apply_report.get("protocol_id"),
        "compile_ok": apply_report.get("compile_ok"),
        "reject": apply_report.get("reject_reason"),
        "monty": (monty.get("summary") or {}),
        "bytes": apply_report.get("content_hash"),
    }


def _payloads() -> list[tuple[str, str]]:
    return [
        ("fixture_rush", (CANDIDATES / "fixture_rush.py").read_text(encoding="utf-8")),
    ]


def main() -> int:
    stack = start_stack()
    try:
        program = httpx.get(f"{stack.orch_url}/program", timeout=10).json()
        print(json.dumps({
            "gepa": stack.orch_url,
            "program_id": program.get("program_id"),
            "apply": "POST /rollout lever_bundle.values.policy_script whole_file.v1",
        }, indent=2))
        rows = []
        winner = None
        for name, source in _payloads():
            body = httpx.post(
                f"{stack.orch_url}/rollout",
                json=_gepa_whole_file(source),
                timeout=180,
            ).json()
            row = _summarize(name, body)
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
            if float(row.get("reward") or 0) > 0.1:
                winner = row
                break
        ranked = sorted(rows, key=lambda r: float(r.get("reward") or 0), reverse=True)
        print(json.dumps({
            "winner": winner,
            "ranked": [{"library": r["library"], "reward": r["reward"], "achievements": r["achievements"], "status": r["status"]} for r in ranked],
        }, indent=2))
        return 0 if winner else 1
    finally:
        stack.stop()


if __name__ == "__main__":
    sys.exit(main())
