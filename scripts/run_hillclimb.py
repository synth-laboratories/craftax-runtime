#!/usr/bin/env python3
"""Rank Craftax heuristic_policy candidates on a fixed achievement suite."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


TASK_DIR = Path(__file__).resolve().parents[1]
if str(TASK_DIR / "scripts") not in sys.path:
    sys.path.insert(0, str(TASK_DIR / "scripts"))

from run_policy_sweep import CandidateEpisodeTimeout, CandidatePolicyFailure, run_policy_sweep


def candidate_paths(candidate_root: Path) -> list[tuple[str, Path]]:
    if not candidate_root.exists():
        return []
    env_root = candidate_root / "craftax"
    search_root = env_root if env_root.is_dir() else candidate_root
    return [(path.parent.name, path) for path in sorted(search_root.glob("*/heuristic_policy.py"))]


def _failure_row(
    candidate_id: str, policy_path: Path, mode: str = "candidate_policy_failure"
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "policy_path": str(policy_path),
        "policy_sha256": "",
        "score": 0.0,
        "mean_reward": 0.0,
        "success_rate": 0.0,
        "mean_achievement_count": 0.0,
        "achievement_frequency": {},
        "invalid_action_count": 1,
        "failure_mode_counts": {mode: 1},
    }


# A candidate that fails to load and a candidate that never returns are both the
# CANDIDATE's fault, and neither should be able to take the grade down with it.
# A single runaway episode used to propagate out of the thread pool and abort the
# whole sweep -- including the baseline's score -- so one pathological submission
# made the run ungradeable instead of scoring itself badly. That is a denial of
# service dressed as an infrastructure failure.
#
# The baseline is exempt by construction: it is graded in the same loop, but a
# baseline that times out is an environment problem, and the typed
# `baseline_policy_failed` exit below still fires because no report is recorded
# for it.
_CANDIDATE_FAULTS = (CandidatePolicyFailure, CandidateEpisodeTimeout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Craftax policy hillclimb leaderboard.")
    # The generated suite is 100 distinct worlds from the engine's own
    # `generate_world_layout`; `policy_dev_v1` is 10 seeds at 80 steps, which is
    # a smoke test, not a measurement.
    parser.add_argument(
        "--suite",
        default=str(TASK_DIR / "defaults" / "policy_sweep" / "craftax_gen_train_v1.json"),
    )
    parser.add_argument(
        "--heldout-suite",
        default="",
        help="optional second suite the candidate never sees; when given, ranking uses "
             "the train suite and the reported score is the held-out one",
    )
    # The graded baseline is the container copy -- the one the sandbox actually
    # loads and the one AGENTS.md points the agent at. Defaulting to
    # `policies/heuristic_baseline.py` meant a local run ranked against a
    # different file than the verifier did, and that file is not self-contained.
    parser.add_argument(
        "--baseline",
        default=str(TASK_DIR / "containers" / "codepolicy" / "heuristic_policy.py"),
    )
    parser.add_argument("--candidate-root", default=str(TASK_DIR / "candidates"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--parallel", type=int, default=1,
                        help="episodes to run concurrently; each holds a sandbox container")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument(
        "--episode-timeout-seconds",
        type=float,
        default=180.0,
        help=(
            "per-episode wall clock in the isolated lane. 30s was sized for the "
            "80-step dev suite; a 500-step generated world graded in its own "
            "container under N-way contention needs more, and an episode killed "
            "at the deadline fails the whole sweep."
        ),
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    suite_path = Path(args.suite).expanduser().resolve()
    baseline_path = Path(args.baseline).expanduser().resolve()
    heldout_path = (
        Path(args.heldout_suite).expanduser().resolve() if args.heldout_suite else None
    )
    submitted_candidates = candidate_paths(
        Path(args.candidate_root).expanduser().resolve()
    )
    baseline_bytes = baseline_path.read_bytes()
    for candidate_id, policy_path in submitted_candidates:
        if candidate_id == "baseline":
            raise SystemExit("candidate id 'baseline' is reserved")
        if policy_path.read_bytes() == baseline_bytes:
            raise SystemExit(
                f"candidate {candidate_id!r} is an exact baseline copy"
            )
    policies: list[tuple[str, Path]] = [("baseline", baseline_path)]
    policies.extend(submitted_candidates)
    if not policies:
        raise SystemExit("no policies found")

    rankings: list[dict[str, Any]] = []
    reports: dict[str, dict[str, Any]] = {}
    heldout_reports: dict[str, dict[str, Any] | None] = {}

    def sweep(policy_path: Path, suite: Path, output_path: Path, *, trace: bool) -> dict[str, Any]:
        return run_policy_sweep(
            policy_path=policy_path,
            suite_path=suite,
            output_path=output_path,
            include_trace=trace,
            parallel=max(1, int(args.parallel)),
            summary_only=bool(args.summary_only),
            # Selects the isolated lane. Without it every policy -- the
            # baseline included -- runs in-process, which is not the rule
            # candidates are graded under, and a baseline that could never
            # load under the sandbox scores as though it had.
            episode_timeout_seconds=float(args.episode_timeout_seconds),
        )

    for candidate_id, policy_path in policies:
        candidate_dir = output / "candidates" / candidate_id
        try:
            report = sweep(
                policy_path,
                suite_path,
                candidate_dir / "summary.json",
                trace=bool(args.include_trace),
            )
        except CandidatePolicyFailure:
            rankings.append(_failure_row(candidate_id, policy_path))
            continue
        except CandidateEpisodeTimeout:
            rankings.append(
                _failure_row(candidate_id, policy_path, "candidate_episode_timeout")
            )
            continue
        reports[candidate_id] = report
        heldout = None
        if heldout_path is not None:
            try:
                heldout = sweep(
                    policy_path,
                    heldout_path,
                    candidate_dir / "heldout_summary.json",
                    trace=False,
                )
            except (CandidatePolicyFailure, CandidateEpisodeTimeout):
                heldout = None
            heldout_reports[candidate_id] = heldout
        rankings.append(
            {
                "candidate_id": candidate_id,
                "policy_path": str(policy_path),
                "policy_sha256": report["policy_sha256"],
                "score": report["score"],
                "mean_reward": report["mean_reward"],
                "success_rate": report.get("success_rate", 0.0),
                "mean_achievement_count": report.get("mean_achievement_count", 0.0),
                "achievement_frequency": report["achievement_frequency"],
                "invalid_action_count": report["invalid_action_count"],
                "failure_mode_counts": report["failure_mode_counts"],
                # Ranking uses train; the reported score is held-out. Without
                # the split, "more seeds" is train-on-test and memorising the
                # suite is a winning strategy.
                **(
                    {
                        "mean_train_score": report["score"],
                        "mean_train_reward": report["mean_reward"],
                        "mean_heldout_score": (
                            float(heldout["score"]) if heldout else 0.0
                        ),
                        "mean_heldout_reward": (
                            float(heldout["mean_reward"]) if heldout else 0.0
                        ),
                        "heldout_success_rate": (
                            float(heldout.get("success_rate", 0.0)) if heldout else 0.0
                        ),
                        "score": float(heldout["score"]) if heldout else 0.0,
                    }
                    if heldout_path is not None
                    else {}
                ),
            }
        )

    if "baseline" not in reports:
        # The baseline is the ranking origin: without it no delta is meaningful.
        # It fails the same way a candidate does -- most often because the policy
        # file is not self-contained, which the sandbox rejects at load.
        raise SystemExit(
            "baseline_policy_failed: the reference policy did not complete the "
            "suite; see the baseline row in the leaderboard rankings"
        )
    baseline_row = next(r for r in rankings if r["candidate_id"] == "baseline")
    baseline_score = float(baseline_row["score"])
    # Rank on the train number even when the reported score is held-out: the
    # held-out suite is a measurement, not a selection signal.
    rankings.sort(
        key=lambda item: (
            float(item.get("mean_train_score", item["score"])),
            float(item.get("mean_train_reward", item["mean_reward"])),
            item["candidate_id"],
        ),
        reverse=True,
    )
    best = rankings[0]
    best_dir = output / "best_policy"
    best_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(best["policy_path"]), best_dir / "heuristic_policy.py")
    leaderboard = {
        "schema": "gamebench.hillclimb.v1",
        "env_family": "craftax-singleplayer",
        "suite_id": reports["baseline"]["suite_id"],
        **(
            {
                "heldout_suite_id": (
                    (heldout_reports.get("baseline") or {}).get("suite_id", "")
                ),
                "train_n_scenarios": int(reports["baseline"]["n_seeds"]),
                "heldout_n_scenarios": int(
                    (heldout_reports.get("baseline") or {}).get("n_seeds", 0)
                ),
            }
            if heldout_path is not None
            else {}
        ),
        "score_metric": "achievement_success_score",
        "policy_isolation": reports["baseline"].get("policy_isolation", "unknown"),
        "policy_isolation_receipt": reports["baseline"].get("policy_isolation_receipt", {}),
        "baseline_score": baseline_score,
        "best_score": float(best["score"]),
        "best_candidate_id": best["candidate_id"],
        "evaluated_policy_count": len(rankings),
        "rankings": [
            {
                **item,
                "delta_vs_baseline": float(item["score"]) - baseline_score,
                "status": "accepted" if item["candidate_id"] == best["candidate_id"] else "rejected",
                "accepted": item["candidate_id"] == best["candidate_id"],
            }
            for item in rankings
        ],
    }
    (output / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2, sort_keys=True) + "\n")
    print(json.dumps(leaderboard, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
