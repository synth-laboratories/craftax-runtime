"""Every action the model asked for is accounted for, or named as dropped.

Seed 202 of the five-chat Craftax review declared eleven actions and executed
ten. Nothing in the proposal record, the trace, or the turn summary said so, so
the raw assistant text and the executed trajectory simply disagreed and a reader
had to guess whether it was a cap, a parser bug, or a bad action name.

It was none of those: `make_torch` is not in this environment's action set, and
it was discarded during resolution without a word.
"""

from __future__ import annotations

import json

from containers.react.action_parsing import parse_actions_text


def test_an_action_this_environment_lacks_is_named_not_dropped_in_silence() -> None:
    seed_202_call_2 = json.dumps(
        {
            "actions": [
                "do", "left", "do", "right", "do",
                "up", "do", "down", "do", "noop",
                "make_torch",
            ]
        }
    )
    parsed = parse_actions_text(
        seed_202_call_2, None, min_actions=1, max_actions=10, steps_remaining=20
    )
    assert parsed.declared_count == 11
    assert parsed.accepted_count == 10
    assert parsed.rejected == (("make_torch", "unknown_action"),)
    # This particular disagreement was never the batch cap, which is exactly why
    # it needed its own reason rather than a shared "truncated" flag.
    assert parsed.truncation_reason is None
    assert parsed.dropped == ()


def test_the_batch_cap_records_what_it_cut_and_why() -> None:
    parsed = parse_actions_text(
        json.dumps({"actions": ["do", "left", "right", "up", "down"]}),
        None,
        min_actions=1,
        max_actions=3,
        steps_remaining=20,
    )
    assert parsed.declared_count == 5
    assert parsed.accepted_count == 3
    assert parsed.dropped == ("up", "down")
    assert parsed.truncation_reason == "batch_cap"


def test_the_step_budget_is_a_different_reason_from_the_batch_cap() -> None:
    parsed = parse_actions_text(
        json.dumps({"actions": ["do", "left", "right", "up", "down"]}),
        None,
        min_actions=1,
        max_actions=10,
        steps_remaining=2,
    )
    assert parsed.accepted_count == 2
    assert parsed.dropped == ("right", "up", "down")
    assert parsed.truncation_reason == "steps_remaining"


def test_a_plan_that_fits_reports_no_drops_at_all() -> None:
    parsed = parse_actions_text(
        json.dumps({"actions": ["do", "left"]}),
        None,
        min_actions=1,
        max_actions=10,
        steps_remaining=20,
    )
    assert parsed.declared_count == parsed.accepted_count == 2
    assert parsed.dropped == ()
    assert parsed.rejected == ()
    assert parsed.truncation_reason is None
