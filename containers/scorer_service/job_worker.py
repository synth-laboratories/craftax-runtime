"""Trusted per-job worker; untrusted policy code remains inside Linux bwrap."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import stat
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import field_validator, model_validator


TASK_ROOT = Path(__file__).resolve().parents[2]
for path in (TASK_ROOT, TASK_ROOT / "gold_python", TASK_ROOT / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from containers.codepolicy.policy_subprocess import CandidatePolicyFailure  # noqa: E402
from containers.scorer_service.contract import StrictModel, require_identifier  # noqa: E402
from scripts.run_policy_sweep import (  # noqa: E402
    CandidateEpisodeTimeout,
    EpisodeWorkerFailure,
    _terminate_active_episode_processes,
    run_policy_sweep,
)


class WorkerRequest(StrictModel):
    schema_version: Literal["gamebench.craftax.score_worker_request.v1"]
    job_id: str
    workspace_directory: str
    policy_path: str
    report_path: str
    suite_id: str
    task_template: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    episode_parallelism: int
    timeout_seconds: float

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_request(self) -> WorkerRequest:
        require_identifier(self.job_id, "job_id")
        require_identifier(self.suite_id, "suite_id")
        workspace = Path(self.workspace_directory)
        policy = Path(self.policy_path)
        report = Path(self.report_path)
        if not workspace.is_absolute() or not policy.is_absolute() or not report.is_absolute():
            raise ValueError("worker paths must be absolute")
        resolved_workspace = workspace.resolve()
        if (
            not policy.resolve().is_relative_to(resolved_workspace)
            or not report.resolve().is_relative_to(resolved_workspace)
        ):
            raise ValueError("worker paths must stay inside the job workspace")
        if not policy.is_file():
            raise ValueError("worker policy file is missing")
        task_path = TASK_ROOT / self.task_template
        if (
            self.task_template.startswith("/")
            or ".." in self.task_template.split("/")
            or not task_path.is_file()
        ):
            raise ValueError("worker task_template is not a tracked task file")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("worker seeds must be non-empty and unique")
        if self.episode_parallelism != 1:
            raise ValueError("worker episode_parallelism must be exactly one")
        if self.max_steps <= 0 or self.timeout_seconds <= 0:
            raise ValueError("worker limits must be positive")
        return self


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _trusted_failure(*, status: str, code: str, origin: str, retryable: bool) -> dict[str, Any]:
    return {
        "schema_version": "gamebench.craftax.score_worker_result.v1",
        "status": status,
        "report_path": None,
        "failure": {
            "code": code,
            "origin": origin,
            "retryable": retryable,
        },
    }


def run(request: WorkerRequest) -> dict[str, Any]:
    report_path = Path(request.report_path)
    suite = {
        "schema": "gamebench.craftax.policy_sweep.v1",
        "suite_id": request.suite_id,
        "task_template": request.task_template,
        "seeds": list(request.seeds),
        "max_steps": request.max_steps,
    }
    try:
        run_policy_sweep(
            policy_path=Path(request.policy_path),
            suite_path=None,
            output_path=report_path,
            suite_payload=suite,
            include_trace=False,
            lane=request.lane,
            base_url="disabled://no-network-rust-repl",
            parallel=request.episode_parallelism,
            summary_only=True,
            progress=None,
            max_steps_override=request.max_steps,
            episode_timeout_seconds=request.timeout_seconds,
        )
    except CandidatePolicyFailure:
        return _trusted_failure(
            status="failed",
            code="candidate_policy_failure",
            origin="candidate",
            retryable=False,
        )
    except CandidateEpisodeTimeout:
        return _trusted_failure(
            status="timed_out",
            code="candidate_episode_timeout",
            origin="trusted_supervisor",
            retryable=False,
        )
    except EpisodeWorkerFailure:
        return _trusted_failure(
            status="failed",
            code="episode_worker_failure",
            origin="trusted_scorer",
            retryable=True,
        )
    except Exception:
        return _trusted_failure(
            status="failed",
            code="scorer_infrastructure_failure",
            origin="trusted_scorer",
            retryable=True,
        )
    return {
        "schema_version": "gamebench.craftax.score_worker_result.v1",
        "status": "succeeded",
        "report_path": str(report_path),
        "failure": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one explicitly authorized isolated Craftax score job"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--start-gate-fd", required=True, type=int)
    args = parser.parse_args()
    try:
        gate_stat = os.fstat(args.start_gate_fd)
        if not stat.S_ISFIFO(gate_stat.st_mode):
            raise RuntimeError("start gate is not a pipe")
        gate_value = os.read(args.start_gate_fd, 2)
    except (OSError, RuntimeError):
        raise SystemExit("worker start gate is invalid") from None
    finally:
        with contextlib.suppress(OSError):
            os.close(args.start_gate_fd)
    if gate_value != b"G":
        raise SystemExit("worker start gate did not authorize execution")
    input_path = Path(args.input)
    if not input_path.is_absolute() or not input_path.is_file():
        raise SystemExit("--input must be an existing absolute file")
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        request = WorkerRequest.model_validate(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        raise SystemExit("worker input is invalid") from None
    result_path = Path(request.workspace_directory) / "worker-result.json"

    def terminate(_signum: int, _frame: Any) -> None:
        _terminate_active_episode_processes()
        raise SystemExit(143)

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        result = run(request)
        _write_result(result_path, result)
        return 0 if result["status"] == "succeeded" else 40
    finally:
        _terminate_active_episode_processes()


if __name__ == "__main__":
    raise SystemExit(main())
