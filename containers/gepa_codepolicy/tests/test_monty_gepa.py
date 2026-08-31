from __future__ import annotations

from pathlib import Path

from containers.codepolicy.monty import MontyLoadError, normalize_spec
from containers.gepa_codepolicy.apply import apply_source, sha256_text
from containers.gepa_codepolicy.process_policy import ProcessPolicy
from containers.gepa_codepolicy.score import episode_reward, universe_size

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_normalize_monty_spec() -> None:
    path = FIXTURES / "noop_policy.py"
    spec = normalize_spec({"kind": "monty_python", "module": "noop", "entry": "choose_actions"}, path=path)
    assert spec["entry"] == "choose_actions"
    assert spec["kind"] == "monty_python"


def test_unknown_monty_kind() -> None:
    path = FIXTURES / "noop_policy.py"
    try:
        normalize_spec({"kind": "registry"}, path=path)
    except MontyLoadError:
        return
    raise AssertionError("expected MontyLoadError")


def test_whole_file_hash() -> None:
    source, report, protocol = apply_source("old", {"protocol_id": "whole_file.v1", "content": "new"})
    assert source == "new"
    assert protocol == "whole_file.v1"
    assert report["content_hash"] == sha256_text("new")


def test_unknown_protocol_fails_closed() -> None:
    source, report, protocol = apply_source("old", {"protocol_id": "ssh.v1", "content": "x"})
    assert source == "old"
    assert report["patch_ok"] is False
    assert "unknown_protocol" in str(report.get("reject_reason"))


def test_process_policy_choose_actions() -> None:
    child = ProcessPolicy(FIXTURES / "noop_policy.py")
    try:
        decision = child(
            observation_text="",
            session={"ply": 0, "lane": "rust"},
            valid_actions=["noop", "do"],
            readout={"observation": {}},
            ply=0,
        )
        assert decision["actions"] == ["noop"]
    finally:
        child.close()


def test_broken_policy_fails_load() -> None:
    broken = FIXTURES / "_broken_tmp.py"
    broken.write_text("def nope():\n    return 1\n", encoding="utf-8")
    try:
        from containers.codepolicy.policy_subprocess import CandidatePolicyFailure

        raised = False
        try:
            ProcessPolicy(broken)
        except CandidatePolicyFailure:
            raised = True
        assert raised
    finally:
        broken.unlink(missing_ok=True)


def test_episode_reward_scale() -> None:
    assert universe_size() > 1
    assert episode_reward([]) == 0.0
    assert episode_reward(["collect_wood"]) == episode_reward(["collect_wood", "collect_wood"])
    assert episode_reward(["collect_wood", "place_table"]) > episode_reward(["collect_wood"])
