from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_policy_sweep.py"
SPEC = importlib.util.spec_from_file_location("craftax_policy_sweep", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


def test_evaluator_owned_rollout_event_is_flushed(tmp_path: Path) -> None:
    journal = tmp_path / "rollout-events.jsonl"
    previous = os.environ.get("GAMEBENCH_ROLLOUT_EVENT_PATH")
    os.environ["GAMEBENCH_ROLLOUT_EVENT_PATH"] = str(journal)
    try:
        SWEEP._emit_rollout_event(
            {
                "schema_version": "synth.trace-stream-event.v1",
                "kind": "environment.step",
                "rollout_id": "trial-a-101",
                "ply": 3,
                "actions": ["up"],
                "reward_total": 1.0,
            }
        )
    finally:
        if previous is None:
            os.environ.pop("GAMEBENCH_ROLLOUT_EVENT_PATH", None)
        else:
            os.environ["GAMEBENCH_ROLLOUT_EVENT_PATH"] = previous

    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert rows == [
        {
            "schema_version": "synth.trace-stream-event.v1",
            "kind": "environment.step",
            "rollout_id": "trial-a-101",
            "ply": 3,
            "actions": ["up"],
            "reward_total": 1.0,
        }
    ]
