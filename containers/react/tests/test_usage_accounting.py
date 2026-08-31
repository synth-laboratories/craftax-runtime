"""Missing usage is missing, and a call that happened is a call that happened.

Nested accounting counted a call only when its usage payload was non-empty, so
five Craftax rollouts that each made three real provider calls reported
`llm_calls: 0` with every token field reading `0` — a number indistinguishable
from a call that genuinely used no tokens. Every rate derived from it was wrong
and nothing said so.

Needs the pinned build-context package the container image installs; skipped
where that is absent.
"""

from __future__ import annotations

import pytest

service = pytest.importorskip(
    "containers.react.craftax_singleplayer_container",
    reason="the Craftax container module needs its pinned synth-containers build",
)


def turn(batch_index: int, usage: dict | None) -> dict:
    return {"batch_index": batch_index, "usage": usage or {}}


def test_a_provider_that_reports_no_usage_leaves_tokens_unavailable() -> None:
    totals = service._turn_usage_totals([turn(0, None), turn(0, None), turn(0, None)])
    # Three calls happened. That is a fact about the calls, not about usage.
    assert totals["llm_calls"] == 3
    assert totals["usage_bearing_calls"] == 0
    # Unavailable, never zero: a zero here reads as "this run used no tokens".
    assert totals["prompt_tokens"] is None
    assert totals["completion_tokens"] is None
    assert totals["total_tokens"] is None
    assert totals["usage_coverage"] == {"reported_by": 0, "of": 3, "complete": False}


def test_partial_usage_reports_what_was_reported_and_says_how_much() -> None:
    totals = service._turn_usage_totals(
        [
            turn(0, {"prompt_tokens": 1000, "completion_tokens": 200}),
            turn(0, None),
        ]
    )
    assert totals["llm_calls"] == 2
    assert totals["usage_bearing_calls"] == 1
    assert totals["prompt_tokens"] == 1000
    assert totals["completion_tokens"] == 200
    assert totals["usage_coverage"]["complete"] is False


def test_a_batched_call_is_one_call_not_one_per_ply() -> None:
    totals = service._turn_usage_totals(
        [
            turn(0, {"prompt_tokens": 900, "completion_tokens": 100}),
            turn(1, None),
            turn(2, None),
        ]
    )
    assert totals["llm_calls"] == 1
    assert totals["usage_bearing_calls"] == 1
    assert totals["usage_coverage"]["complete"] is True
