from __future__ import annotations

import sys
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parents[1]
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from containers.codepolicy.rollout_code_policy import (  # noqa: E402
    policy_stop_reason,
    rollout_code_policy,
)


def test_policy_stop_requires_the_literal_boolean() -> None:
    assert policy_stop_reason({"stop_episode": "yes", "stop_reason": "wrong"}) is None
    assert policy_stop_reason({"stop_episode": True, "stop_reason": "budget exhausted"}) == "budget exhausted"


def test_policy_stop_ends_before_a_fallback_action() -> None:
    calls = 0

    def exhausted_policy(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {
            "actions": ["noop"],
            "stop_episode": True,
            "stop_reason": "call cap reached (20)",
        }

    result = rollout_code_policy(
        policy_path=Path(__file__),
        seed=101,
        max_steps=5,
        candidate_fn=exhausted_policy,
    )

    details = result["reward_info"]["details"]
    assert calls == 1
    assert details["steps"] == 0
    assert details["outcome"] == "truncated"
    assert details["policy_requested_stop"] is True
    assert details["policy_stop_reason"] == "call cap reached (20)"
    assert result["artifact"][0]["turns"] == []
