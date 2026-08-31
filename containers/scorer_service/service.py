"""Cloud Slot service for typed, isolated Craftax candidate score jobs."""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import hmac
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import model_validator

from containers.scorer_service.authority import ScorerAuthority, ScoringProfile
from containers.scorer_service.claim_authority import (
    ClaimAuthorityClient,
    ClaimAuthorityError,
)
from containers.scorer_service.contract import (
    CandidateScoreCancellation,
    CandidateScoreCleanupReceipt,
    CandidateScoreIdentity,
    CandidateScoreRequest,
    CandidateScoreResult,
    CandidateScoreSubmission,
    JobActionRequest,
    ScoreFailure,
    ScoreInputAuthority,
    ScoreOutputAuthority,
    ScoreRow,
    StrictModel,
    canonical_json_sha256,
    identity_payload,
    response_payload,
    require_identifier,
    require_sha256,
)


TASK_ROOT = Path(__file__).resolve().parents[2]
JOBS_PATH = "/v1/gamebench/scorers/craftax/jobs"
_STATE_SCHEMA = "gamebench.craftax.score_job_state.v1"
_WORKER_RESULT_SCHEMA = "gamebench.craftax.score_worker_result.v1"
_WORKER_STARTUP_CLEANUP_SECONDS = 30.0


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid trusted scorer JSON artifact: {path.name}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"trusted scorer JSON artifact is not an object: {path.name}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


class ScoreServiceError(RuntimeError):
    def __init__(self, status_code: int, code: str) -> None:
        super().__init__(code)
        self.status_code = status_code
        self.code = code


class PersistedProcessIdentity(StrictModel):
    """PID-reuse-safe identity for one scorer-owned Linux worker group."""

    pid: int
    proc_start_ticks: int
    cmdline_sha256: str
    worker_input_path: str

    @classmethod
    def capture(
        cls,
        *,
        pid: int,
        worker_input_path: Path,
        start_gate_fd: int,
    ) -> PersistedProcessIdentity:
        if isinstance(pid, bool) or pid <= 0:
            raise RuntimeError("worker PID is invalid")
        proc = Path("/proc") / str(pid)
        try:
            stat_bytes = (proc / "stat").read_bytes()
            cmdline = (proc / "cmdline").read_bytes()
        except OSError as exc:
            raise RuntimeError("worker process identity is unreadable") from exc
        start_ticks = _proc_start_ticks(stat_bytes)
        argv = tuple(part.decode("utf-8", errors="strict") for part in cmdline.split(b"\0") if part)
        expected_tail = (
            "-m",
            "containers.scorer_service.job_worker",
            "--input",
            str(worker_input_path),
            "--start-gate-fd",
            str(start_gate_fd),
        )
        if len(argv) < len(expected_tail) + 1 or argv[-len(expected_tail) :] != expected_tail:
            raise RuntimeError("worker process command line is not scorer-owned")
        return cls.model_validate(
            {
                "pid": pid,
                "proc_start_ticks": start_ticks,
                "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
                "worker_input_path": str(worker_input_path),
            }
        )

    @model_validator(mode="after")
    def validate_fields(self) -> PersistedProcessIdentity:
        if isinstance(self.pid, bool) or self.pid <= 0:
            raise ValueError("persisted worker pid must be positive")
        if isinstance(self.proc_start_ticks, bool) or self.proc_start_ticks <= 0:
            raise ValueError("persisted worker proc_start_ticks must be positive")
        require_sha256(self.cmdline_sha256, "cmdline_sha256")
        require_identifier(self.worker_input_path, "worker_input_path")
        if not Path(self.worker_input_path).is_absolute():
            raise ValueError("worker_input_path must be absolute")
        return self

    def is_live_exact(self) -> bool:
        """Return true only while the recorded Linux worker identity still exists."""
        proc = Path("/proc") / str(self.pid)
        try:
            stat_bytes = (proc / "stat").read_bytes()
            cmdline = (proc / "cmdline").read_bytes()
        except OSError:
            return False
        return (
            _proc_start_ticks(stat_bytes) == self.proc_start_ticks
            and hmac.compare_digest(
                hashlib.sha256(cmdline).hexdigest(),
                self.cmdline_sha256,
            )
        )


def _proc_start_ticks(stat_bytes: bytes) -> int:
    closing = stat_bytes.rfind(b")")
    if closing < 0:
        raise RuntimeError("Linux process stat has no command terminator")
    fields = stat_bytes[closing + 1 :].strip().split()
    if len(fields) <= 19:
        raise RuntimeError("Linux process stat is incomplete")
    try:
        start_ticks = int(fields[19])
    except ValueError as exc:
        raise RuntimeError("Linux process starttime is invalid") from exc
    if start_ticks <= 0:
        raise RuntimeError("Linux process starttime must be positive")
    return start_ticks


@dataclass
class _Job:
    identity: CandidateScoreIdentity
    request_body_sha256: str
    candidate_size_bytes: int
    submission: CandidateScoreSubmission
    profile: ScoringProfile | None
    request: CandidateScoreRequest | None
    workspace: Path
    candidate_path: Path
    process: subprocess.Popen[Any] | None = None
    process_identity: PersistedProcessIdentity | None = None
    result: CandidateScoreResult | None = None
    cancellation: CandidateScoreCancellation | None = None
    cleanup: CandidateScoreCleanupReceipt | None = None
    started_at: str | None = None


class ScoreJobManager:
    """Single authority owner for scorer job admission and resource lifecycle."""

    def __init__(self, authority: ScorerAuthority) -> None:
        self.authority = authority
        self._state_directory = Path(authority.state_directory)
        self._workspace_directory = Path(authority.workspace_directory)
        self._claim_authority = ClaimAuthorityClient(
            backend_api_base_url=authority.backend_api_base_url,
            service_auth_file=Path(authority.service_auth_file),
            timeout_seconds=authority.backend_claim_read_timeout_seconds,
            environment=authority.environment,
            cloud_slot=authority.cloud_slot,
            deployment_id=authority.deployment_id,
            claim_id=authority.claim_id,
            fencing_token=authority.fencing_token,
        )
        self._claim_authority.assert_current()
        self._lock = threading.RLock()
        self._jobs: dict[str, _Job] = {}
        self._idempotency_jobs: dict[str, str] = {}
        self._request_jobs: dict[str, str] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=authority.max_active_jobs,
            thread_name_prefix="craftax-score-job",
        )
        recoverable = self._load_state()
        for job in recoverable:
            self._executor.submit(self._execute, job)

    def close(self) -> None:
        """Terminate only scorer-owned worker process groups, then stop dispatch."""
        with self._lock:
            processes = [job.process for job in self._jobs.values() if job.process]
        for process in processes:
            if process is not None and process.poll() is None:
                self._terminate_process(process)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def submit(self, request: CandidateScoreRequest) -> CandidateScoreSubmission:
        try:
            self._claim_authority.assert_current()
        except ClaimAuthorityError as exc:
            raise ScoreServiceError(409, "score_cloud_claim_authority_invalid") from exc
        try:
            profile = self.authority.assert_request(request)
        except (ValueError, RuntimeError) as exc:
            raise ScoreServiceError(422, "score_request_authority_rejected") from exc
        request_hash = request.request_body_sha256()
        with self._lock:
            existing_id = self._idempotency_jobs.get(request.idempotency_key)
            if existing_id is not None:
                existing = self._jobs[existing_id]
                if existing.request_body_sha256 != request_hash:
                    raise ScoreServiceError(409, "score_idempotency_conflict")
                return existing.submission
            request_job_id = self._request_jobs.get(request.request_id)
            if request_job_id is not None:
                existing = self._jobs[request_job_id]
                if existing.request_body_sha256 != request_hash:
                    raise ScoreServiceError(409, "score_request_id_conflict")
                return existing.submission
            retained = sum(job.cleanup is None for job in self._jobs.values())
            pending = sum(job.result is None for job in self._jobs.values())
            if len(self._jobs) >= self.authority.max_cleaned_state_records:
                raise ScoreServiceError(429, "score_state_record_capacity_exhausted")
            if retained >= self.authority.max_retained_jobs:
                raise ScoreServiceError(429, "score_retained_job_capacity_exhausted")
            if pending >= self.authority.max_active_jobs + self.authority.max_queued_jobs:
                raise ScoreServiceError(429, "score_execution_capacity_exhausted")

            job_id = f"gbscore_{uuid.uuid4().hex}"
            workspace = self._workspace_directory / job_id
            workspace.mkdir(mode=0o700)
            input_root = workspace / "input"
            input_root.mkdir(mode=0o700)
            candidate_path = input_root / request.entrypoint
            candidate_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            candidate_path.write_bytes(request.decoded_candidate_bytes())
            candidate_path.chmod(0o400)
            created_at = _utc_now()
            input_authority = ScoreInputAuthority.model_validate(
                {
                    "schema_version": "gamebench.craftax.score_input_authority.v1",
                    "request_body_sha256": request_hash,
                    "candidate_sha256": request.candidate_sha256,
                    "candidate_size_bytes": len(request.decoded_candidate_bytes()),
                    "idempotency_key": request.idempotency_key,
                }
            )
            submission = CandidateScoreSubmission.model_validate(
                {
                    **identity_payload(request),
                    "schema_version": "gamebench.craftax.score_submission.v1",
                    "job_id": job_id,
                    "status": "accepted",
                    "created_at": created_at,
                    "input_authority": response_payload(input_authority),
                }
            )
            identity = CandidateScoreIdentity.model_validate(identity_payload(request))
            job = _Job(
                identity=identity,
                request_body_sha256=request_hash,
                candidate_size_bytes=len(request.decoded_candidate_bytes()),
                submission=submission,
                profile=profile,
                request=request,
                workspace=workspace,
                candidate_path=candidate_path,
            )
            self._jobs[job_id] = job
            self._idempotency_jobs[request.idempotency_key] = job_id
            self._request_jobs[request.request_id] = job_id
            self._persist(job)
            self._executor.submit(self._execute, job)
            return submission

    def get(
        self, job_id: str
    ) -> CandidateScoreSubmission | CandidateScoreResult:
        with self._lock:
            job = self._required_job(job_id)
            return job.result or job.submission

    def cancel(self, job_id: str) -> CandidateScoreCancellation:
        with self._lock:
            job = self._required_job(job_id)
            if job.cancellation is not None:
                return job.cancellation
            if job.result is not None:
                cancellation = CandidateScoreCancellation.model_validate(
                    {
                        **job.identity.model_dump(mode="json"),
                        "schema_version": "gamebench.craftax.score_cancellation.v1",
                        "job_id": job_id,
                        "result_id": job.result.result_id,
                        "status": "already_terminal",
                        "process_group_terminated": self._process_is_terminated(job.process),
                        "cleanup_required": job.cleanup is None,
                        "cancelled_at": _utc_now(),
                    }
                )
                job.cancellation = cancellation
                self._persist(job)
                return cancellation
            if job.process is not None and job.process.poll() is None:
                self._terminate_process(job.process)
            if not self._process_is_terminated(job.process):
                raise ScoreServiceError(500, "score_process_group_termination_unproved")
            started_at = job.started_at or job.submission.created_at
            job.result = self._failure_result(
                job,
                status="cancelled",
                code="score_job_cancelled",
                origin="trusted_supervisor",
                retryable=False,
                started_at=started_at,
            )
            cancellation = CandidateScoreCancellation.model_validate(
                {
                    **job.identity.model_dump(mode="json"),
                    "schema_version": "gamebench.craftax.score_cancellation.v1",
                    "job_id": job_id,
                    "result_id": job.result.result_id,
                    "status": "cancelled",
                    "process_group_terminated": True,
                    "cleanup_required": True,
                    "cancelled_at": _utc_now(),
                }
            )
            job.cancellation = cancellation
            self._persist(job)
            return cancellation

    def cleanup(self, job_id: str) -> CandidateScoreCleanupReceipt:
        with self._lock:
            job = self._required_job(job_id)
            if job.cleanup is not None:
                return job.cleanup
            if job.result is None:
                raise ScoreServiceError(409, "score_cleanup_requires_terminal_job")
            if job.process is not None and job.process.poll() is None:
                self._terminate_process(job.process)
            if not self._process_is_terminated(job.process):
                raise ScoreServiceError(500, "score_cleanup_process_residue")
            if job.workspace.exists():
                shutil.rmtree(job.workspace)
            workspace_removed = not job.workspace.exists()
            if not workspace_removed:
                raise ScoreServiceError(500, "score_cleanup_workspace_residue")
            receipt = CandidateScoreCleanupReceipt.model_validate(
                {
                    **job.identity.model_dump(mode="json"),
                    "schema_version": "gamebench.craftax.score_cleanup.v1",
                    "job_id": job_id,
                    "result_id": job.result.result_id,
                    "status": "cleaned",
                    "workspace_removed": True,
                    "process_group_terminated": True,
                    "policy_sandboxes_removed": True,
                    "processes_remaining": 0,
                    "containers_remaining": 0,
                    "resources_remaining": 0,
                    "state_record_disposition": "retained_until_external_archive",
                    "cleaned_at": _utc_now(),
                }
            )
            job.cleanup = receipt
            job.request = None
            job.profile = None
            self._persist(job)
            return receipt

    def health(self) -> dict[str, Any]:
        try:
            self._claim_authority.assert_current()
            claim_current = True
        except ClaimAuthorityError:
            claim_current = False
        with self._lock:
            active = sum(
                job.result is None and job.process is not None and job.process.poll() is None
                for job in self._jobs.values()
            )
            queued = sum(job.result is None and job.process is None for job in self._jobs.values())
            residue = sum(job.result is not None and job.cleanup is None for job in self._jobs.values())
        return {
            "schema_version": "gamebench.craftax.scorer_health.v1",
            "status": "healthy" if claim_current else "unhealthy",
            "claim_current": claim_current,
            "environment": self.authority.environment,
            "cloud_slot": self.authority.cloud_slot,
            "deployment_id": self.authority.deployment_id,
            "claim_id": self.authority.claim_id,
            "fencing_token": self.authority.fencing_token,
            "gamebench_source_sha": self.authority.gamebench_source_sha,
            "scorer_source_sha": self.authority.scorer_source_sha,
            "scorer_image_digest": self.authority.scorer_image_digest,
            "active_jobs": active,
            "queued_jobs": queued,
            "terminal_jobs_awaiting_cleanup": residue,
        }

    def info(self) -> dict[str, Any]:
        profile_authority = [
            profile.model_dump(mode="json") for profile in self.authority.profiles
        ]
        return {
            "schema_version": "gamebench.craftax.scorer_info.v1",
            "service": "craftax_candidate_scorer",
            "lane": "rust",
            "candidate_execution": "linux_bwrap_no_network",
            "environment_allowlist": [],
            "profile_authority_sha256": canonical_json_sha256(profile_authority),
            "profile_count": len(profile_authority),
            "scorer_fixture_manifest_sha256": (
                self.authority.scorer_fixture_manifest_sha256
            ),
            "scorer_binary_sha256": self.authority.scorer_binary_sha256,
        }

    def _execute(self, job: _Job) -> None:
        with self._lock:
            if job.result is not None:
                return
            if job.request is None or job.profile is None:
                job.result = self._failure_result(
                    job,
                    status="failed",
                    code="scorer_restart_candidate_missing",
                    origin="trusted_scorer",
                    retryable=False,
                    started_at=job.submission.created_at,
                )
                self._persist(job)
                return
            try:
                self._claim_authority.assert_current()
            except ClaimAuthorityError:
                job.result = self._failure_result(
                    job,
                    status="failed",
                    code="score_cloud_claim_authority_lost_before_start",
                    origin="trusted_scorer",
                    retryable=False,
                    started_at=job.submission.created_at,
                )
                self._persist(job)
                return
            job.started_at = _utc_now()
            worker_input = job.workspace / "worker-input.json"
            report_path = job.workspace / "score-report.json"
            process: subprocess.Popen[str] | None = None
            start_gate_read_fd = -1
            start_gate_write_fd = -1
            try:
                _atomic_json(
                    worker_input,
                    {
                        "schema_version": "gamebench.craftax.score_worker_request.v1",
                        "job_id": job.submission.job_id,
                        "workspace_directory": str(job.workspace),
                        "policy_path": str(job.candidate_path),
                        "report_path": str(report_path),
                        "suite_id": job.request.suite_id,
                        "task_template": job.profile.task_template,
                        "seeds": list(job.request.seeds),
                        "max_steps": job.request.max_steps,
                        "lane": job.request.lane,
                        "episode_parallelism": self.authority.episode_parallelism,
                        "timeout_seconds": job.request.timeout_seconds,
                    },
                )
                attempt_id = uuid.uuid4().hex
                worker_home = job.workspace / f"home-{attempt_id}"
                worker_tmp = job.workspace / f"tmp-{attempt_id}"
                worker_home.mkdir(mode=0o700)
                worker_tmp.mkdir(mode=0o700)
                worker_env = {
                    "HOME": str(worker_home),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "PYTHONPATH": str(TASK_ROOT),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "TMPDIR": str(worker_tmp),
                }
                start_gate_read_fd, start_gate_write_fd = os.pipe()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "containers.scorer_service.job_worker",
                        "--input",
                        str(worker_input),
                        "--start-gate-fd",
                        str(start_gate_read_fd),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=TASK_ROOT,
                    env=worker_env,
                    close_fds=True,
                    pass_fds=(start_gate_read_fd,),
                    start_new_session=True,
                )
                os.close(start_gate_read_fd)
                inherited_start_gate_fd = start_gate_read_fd
                start_gate_read_fd = -1
                job.process = process
                job.process_identity = PersistedProcessIdentity.capture(
                    pid=process.pid,
                    worker_input_path=worker_input,
                    start_gate_fd=inherited_start_gate_fd,
                )
                self._persist(job)
                if os.write(start_gate_write_fd, b"G") != 1:
                    raise RuntimeError("scorer worker start gate write was incomplete")
            except Exception:
                if process is not None:
                    self._terminate_process(process)
                job.result = self._failure_result(
                    job,
                    status="failed",
                    code="score_worker_start_failed",
                    origin="trusted_scorer",
                    retryable=True,
                    started_at=job.started_at,
                )
                self._persist(job)
                return
            finally:
                for descriptor in (start_gate_read_fd, start_gate_write_fd):
                    if descriptor >= 0:
                        with contextlib.suppress(OSError):
                            os.close(descriptor)
            self._persist(job)
            episode_batches = math.ceil(
                len(job.request.seeds) / job.request.episode_parallelism
            )
            timeout_seconds = (
                job.request.timeout_seconds * episode_batches
                + _WORKER_STARTUP_CLEANUP_SECONDS
            )

        timed_out = False
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._terminate_process(process)
        else:
            # Episode workers intentionally share this exact job process group so
            # a scorer crash cannot orphan candidate descendants.  The leader can
            # exit after reporting an episode timeout while a wedged bwrap/policy
            # descendant remains in the group, so sweep the group even after a
            # normal wait before accepting any terminal worker result.
            self._terminate_process(process)

        with self._lock:
            if job.result is not None:
                return
            if timed_out:
                job.result = self._failure_result(
                    job,
                    status="timed_out",
                    code="score_job_timeout",
                    origin="trusted_supervisor",
                    retryable=False,
                    started_at=job.started_at or job.submission.created_at,
                )
                self._persist(job)
                return
            job.result = self._result_from_worker(job)
            self._persist(job)

    def _result_from_worker(self, job: _Job) -> CandidateScoreResult:
        result_path = job.workspace / "worker-result.json"
        if not result_path.is_file():
            return self._failure_result(
                job,
                status="failed",
                code="score_worker_result_missing",
                origin="trusted_scorer",
                retryable=True,
                started_at=job.started_at or job.submission.created_at,
            )
        worker = _strict_json(result_path)
        if set(worker) != {"schema_version", "status", "report_path", "failure"}:
            return self._failure_result(
                job,
                status="failed",
                code="score_worker_result_schema_invalid",
                origin="trusted_scorer",
                retryable=True,
                started_at=job.started_at or job.submission.created_at,
            )
        if worker["schema_version"] != _WORKER_RESULT_SCHEMA:
            return self._failure_result(
                job,
                status="failed",
                code="score_worker_result_version_invalid",
                origin="trusted_scorer",
                retryable=True,
                started_at=job.started_at or job.submission.created_at,
            )
        if worker["status"] != "succeeded":
            failure = worker.get("failure")
            if not isinstance(failure, dict) or set(failure) != {
                "code",
                "origin",
                "retryable",
            }:
                return self._failure_result(
                    job,
                    status="failed",
                    code="score_worker_failure_schema_invalid",
                    origin="trusted_scorer",
                    retryable=True,
                    started_at=job.started_at or job.submission.created_at,
                )
            status = worker["status"]
            if status not in {"failed", "timed_out"}:
                status = "failed"
            origin = failure["origin"]
            if origin not in {"candidate", "trusted_supervisor", "trusted_scorer"}:
                origin = "trusted_scorer"
            return self._failure_result(
                job,
                status=status,
                code=str(failure["code"]),
                origin=origin,
                retryable=bool(failure["retryable"]),
                started_at=job.started_at or job.submission.created_at,
            )
        try:
            self._claim_authority.assert_current()
        except ClaimAuthorityError:
            return self._failure_result(
                job,
                status="failed",
                code="score_cloud_claim_authority_lost",
                origin="trusted_scorer",
                retryable=False,
                started_at=job.started_at or job.submission.created_at,
            )
        report_path = job.workspace / "score-report.json"
        if worker["report_path"] != str(report_path) or not report_path.is_file():
            return self._failure_result(
                job,
                status="failed",
                code="score_report_authority_mismatch",
                origin="trusted_scorer",
                retryable=True,
                started_at=job.started_at or job.submission.created_at,
            )
        try:
            return self._success_result(job, _strict_json(report_path))
        except (KeyError, TypeError, ValueError, RuntimeError):
            return self._failure_result(
                job,
                status="failed",
                code="score_report_validation_failed",
                origin="trusted_scorer",
                retryable=True,
                started_at=job.started_at or job.submission.created_at,
            )

    def _success_result(
        self, job: _Job, report: dict[str, Any]
    ) -> CandidateScoreResult:
        request = job.request
        if request is None:
            raise RuntimeError("terminal scorer request identity unavailable")
        exact = {
            "schema": "gamebench.craftax.policy_sweep_summary.v1",
            "env_family": "craftax-singleplayer",
            "suite_id": request.suite_id,
            "suite_path": None,
            "suite_source": "stdin",
            "max_steps": request.max_steps,
            "lane": request.lane,
            "engine_mode": "rust_repl",
            "policy_isolation": "os_sandbox_observation_action.v2",
            "episode_timeout_seconds": request.timeout_seconds,
            "base_url": None,
            "policy_sha256": request.candidate_sha256,
            "seeds": list(request.seeds),
            "n_seeds": len(request.seeds),
        }
        for field, value in exact.items():
            if report.get(field) != value:
                raise ValueError(f"score report mismatch for {field}")
        summaries = report.get("episode_summaries")
        if not isinstance(summaries, list) or len(summaries) != len(request.seeds):
            raise ValueError("score report has incomplete episode rows")
        rows: list[ScoreRow] = []
        for expected_seed, summary in zip(request.seeds, summaries, strict=True):
            if not isinstance(summary, dict) or summary.get("seed") != expected_seed:
                raise ValueError("score report changed seed order")
            isolation = summary.get("policy_isolation")
            if (
                not isinstance(isolation, dict)
                or isolation.get("contract") != "os_sandbox_observation_action.v2"
                or isolation.get("platform") != "linux"
                or isolation.get("sandbox") != "bubblewrap"
                or isolation.get("network") != "unshared"
                or isolation.get("suite_visible") is not False
                or isolation.get("output_visible") is not False
                or isolation.get("evaluator_visible") is not False
                or isolation.get("forking") != "denied"
            ):
                raise ValueError("score report isolation receipt is invalid")
            achievements = summary.get("achievements")
            if not isinstance(achievements, list):
                raise ValueError("score row achievements are invalid")
            row = ScoreRow.model_validate(
                {
                    "seed": expected_seed,
                    "reward": float(summary["reward"]),
                    "achievement_count": int(summary["achievement_count"]),
                    "achievements": achievements,
                    "rollout_id": str(summary.get("rollout_id") or ""),
                    "policy_isolation": "os_sandbox_observation_action.v2",
                    "episode_status": "succeeded",
                }
            )
            rows.append(row)
        score_rows = tuple(rows)
        rows_sha = canonical_json_sha256(
            [response_payload(row) for row in score_rows]
        )
        output_authority = ScoreOutputAuthority.model_validate(
            {
                "schema_version": "gamebench.craftax.score_output_authority.v1",
                "score_rows_sha256": rows_sha,
                "candidate_sha256": request.candidate_sha256,
                "gamebench_source_sha": request.gamebench_source_sha,
                "scorer_source_sha": request.scorer_source_sha,
                "scorer_fixture_manifest_sha256": (
                    request.scorer_fixture_manifest_sha256
                ),
                "scorer_binary_sha256": request.scorer_binary_sha256,
                "scorer_image_digest": request.scorer_image_digest,
                "task_id": request.task_id,
                "suite_id": request.suite_id,
                "seeds": list(request.seeds),
                "max_steps": request.max_steps,
                "lane": request.lane,
                "policy_identity": request.policy_identity,
                "isolation_contract": "os_sandbox_observation_action.v2",
                "network_mode": request.network_mode,
                "environment_allowlist": [],
                "candidate_suite_visible": False,
                "candidate_evaluator_visible": False,
                "terminal": True,
                "environment": request.environment,
                "cloud_slot": request.cloud_slot,
                "deployment_id": request.deployment_id,
                "claim_id": request.claim_id,
                "fencing_token": request.fencing_token,
            }
        )
        mean_reward = float(report["mean_reward"])
        benchmark_score = float(report["score"])
        if not math.isfinite(mean_reward) or not math.isfinite(benchmark_score):
            raise ValueError("score report contains non-finite aggregates")
        return CandidateScoreResult.model_validate(
            {
                **job.identity.model_dump(mode="json"),
                "schema_version": "gamebench.craftax.score_result.v1",
                "job_id": job.submission.job_id,
                "result_id": f"gbresult_{uuid.uuid4().hex}",
                "status": "succeeded",
                "mean_reward": mean_reward,
                "benchmark_score": benchmark_score,
                "score_rows": [response_payload(row) for row in score_rows],
                "output_authority": response_payload(output_authority),
                "failure": None,
                "created_at": job.submission.created_at,
                "started_at": job.started_at or job.submission.created_at,
                "finished_at": _utc_now(),
            }
        )

    def _failure_result(
        self,
        job: _Job,
        *,
        status: str,
        code: str,
        origin: str,
        retryable: bool,
        started_at: str,
    ) -> CandidateScoreResult:
        failure = ScoreFailure.model_validate(
            {"code": code, "origin": origin, "retryable": retryable}
        )
        return CandidateScoreResult.model_validate(
            {
                **job.identity.model_dump(mode="json"),
                "schema_version": "gamebench.craftax.score_result.v1",
                "job_id": job.submission.job_id,
                "result_id": f"gbresult_{uuid.uuid4().hex}",
                "status": status,
                "mean_reward": None,
                "benchmark_score": None,
                "score_rows": [],
                "output_authority": None,
                "failure": response_payload(failure),
                "created_at": job.submission.created_at,
                "started_at": started_at,
                "finished_at": _utc_now(),
            }
        )

    def _terminate_process(self, process: subprocess.Popen[Any]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if process.poll() is None:
            try:
                process.wait(timeout=self.authority.process_termination_grace_seconds)
            except subprocess.TimeoutExpired:
                pass
        deadline = time.monotonic() + self.authority.process_termination_grace_seconds
        while time.monotonic() < deadline:
            if self._process_is_terminated(process):
                return
            time.sleep(0.05)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.authority.process_termination_grace_seconds
        while time.monotonic() < deadline:
            if self._process_is_terminated(process):
                return
            time.sleep(0.05)
        raise RuntimeError("scorer worker process-group termination is unproved")

    @staticmethod
    def _process_is_terminated(process: subprocess.Popen[Any] | None) -> bool:
        if process is None:
            return True
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return True
        return False

    def _required_job(self, job_id: str) -> _Job:
        if not isinstance(job_id, str) or not job_id or job_id != job_id.strip():
            raise ScoreServiceError(422, "score_job_id_invalid")
        job = self._jobs.get(job_id)
        if job is None:
            raise ScoreServiceError(404, "score_job_not_found")
        return job

    def _state_path(self, job_id: str) -> Path:
        return self._state_directory / f"{job_id}.json"

    def _persist(self, job: _Job) -> None:
        payload = {
            "schema_version": _STATE_SCHEMA,
            "identity": job.identity.model_dump(mode="json"),
            "request_body_sha256": job.request_body_sha256,
            "candidate_size_bytes": job.candidate_size_bytes,
            "submission": response_payload(job.submission),
            "result": response_payload(job.result) if job.result else None,
            "cancellation": (
                response_payload(job.cancellation) if job.cancellation else None
            ),
            "cleanup": response_payload(job.cleanup) if job.cleanup else None,
            "workspace_directory": str(job.workspace),
            "candidate_path": str(job.candidate_path),
            "started_at": job.started_at,
            "process_identity": (
                job.process_identity.model_dump(mode="json")
                if job.process_identity is not None
                else None
            ),
        }
        _atomic_json(self._state_path(job.submission.job_id), payload)

    def _terminate_persisted_process(
        self,
        identity: PersistedProcessIdentity,
    ) -> None:
        if not identity.is_live_exact():
            if self._persisted_group_exists(identity):
                raise RuntimeError(
                    "persisted scorer worker identity changed while process group remains"
                )
            return
        try:
            os.killpg(identity.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.authority.process_termination_grace_seconds
        while self._persisted_group_exists(identity) and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._persisted_group_exists(identity):
            try:
                os.killpg(identity.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + self.authority.process_termination_grace_seconds
            while self._persisted_group_exists(identity) and time.monotonic() < deadline:
                time.sleep(0.05)
        if self._persisted_group_exists(identity):
            raise RuntimeError("persisted scorer worker termination is unproved")

    @staticmethod
    def _persisted_group_exists(identity: PersistedProcessIdentity) -> bool:
        try:
            os.killpg(identity.pid, 0)
        except ProcessLookupError:
            return False
        return True

    def _load_state(self) -> list[_Job]:
        recoverable: list[_Job] = []
        expected_keys = {
            "schema_version",
            "identity",
            "request_body_sha256",
            "candidate_size_bytes",
            "submission",
            "result",
            "cancellation",
            "cleanup",
            "workspace_directory",
            "candidate_path",
            "started_at",
            "process_identity",
        }
        for path in sorted(self._state_directory.glob("gbscore_*.json")):
            payload = _strict_json(path)
            if set(payload) != expected_keys or payload["schema_version"] != _STATE_SCHEMA:
                raise RuntimeError(f"invalid persisted score job schema: {path.name}")
            identity = CandidateScoreIdentity.model_validate(payload["identity"])
            submission = CandidateScoreSubmission.model_validate(payload["submission"])
            if submission.job_id != path.stem:
                raise RuntimeError(f"persisted score job path/id mismatch: {path.name}")
            result = (
                CandidateScoreResult.model_validate(payload["result"])
                if payload["result"] is not None
                else None
            )
            cancellation = (
                CandidateScoreCancellation.model_validate(payload["cancellation"])
                if payload["cancellation"] is not None
                else None
            )
            cleanup = (
                CandidateScoreCleanupReceipt.model_validate(payload["cleanup"])
                if payload["cleanup"] is not None
                else None
            )
            process_identity = (
                PersistedProcessIdentity.model_validate(payload["process_identity"])
                if payload["process_identity"] is not None
                else None
            )
            workspace = Path(payload["workspace_directory"])
            candidate_path = Path(payload["candidate_path"])
            if not workspace.is_absolute() or not candidate_path.is_absolute():
                raise RuntimeError(f"persisted score job paths are not absolute: {path.name}")
            if not workspace.resolve().is_relative_to(self._workspace_directory.resolve()):
                raise RuntimeError(f"persisted score workspace escaped authority: {path.name}")
            request: CandidateScoreRequest | None = None
            profile: ScoringProfile | None = None
            if cleanup is None and candidate_path.is_file():
                wire = identity.model_dump(mode="json")
                wire["candidate_bytes"] = base64.b64encode(
                    candidate_path.read_bytes()
                ).decode("ascii")
                request = CandidateScoreRequest.model_validate(wire)
                profile = self.authority.assert_request(request)
                if request.request_body_sha256() != payload["request_body_sha256"]:
                    raise RuntimeError(f"persisted score request hash mismatch: {path.name}")
            job = _Job(
                identity=identity,
                request_body_sha256=str(payload["request_body_sha256"]),
                candidate_size_bytes=int(payload["candidate_size_bytes"]),
                submission=submission,
                profile=profile,
                request=request,
                workspace=workspace,
                candidate_path=candidate_path,
                result=result,
                cancellation=cancellation,
                cleanup=cleanup,
                process_identity=process_identity,
                started_at=(
                    str(payload["started_at"]) if payload["started_at"] is not None else None
                ),
            )
            if submission.idempotency_key in self._idempotency_jobs:
                raise RuntimeError("persisted score jobs contain duplicate idempotency keys")
            if submission.request_id in self._request_jobs:
                raise RuntimeError("persisted score jobs contain duplicate request ids")
            self._jobs[submission.job_id] = job
            self._idempotency_jobs[submission.idempotency_key] = submission.job_id
            self._request_jobs[submission.request_id] = submission.job_id
            if result is None:
                if process_identity is not None:
                    self._terminate_persisted_process(process_identity)
                recoverable.append(job)
        if sum(job.cleanup is None for job in self._jobs.values()) > self.authority.max_retained_jobs:
            raise RuntimeError("persisted score jobs exceed retained-job authority")
        if len(self._jobs) > self.authority.max_cleaned_state_records:
            raise RuntimeError("persisted score jobs exceed state-record authority")
        return recoverable


def create_app(authority: ScorerAuthority) -> FastAPI:
    """Construct the service only after exact runtime authority verification."""
    authority.validate_runtime()
    manager = ScoreJobManager(authority)
    app = FastAPI(
        title="GameBench isolated Craftax candidate scorer",
        version="1.0.0",
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error_code": "score_request_schema_invalid"},
        )

    @app.middleware("http")
    async def require_bounded_body(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith(JOBS_PATH) or request.url.path == "/info":
            authorization = request.headers.get("authorization")
            if (
                not isinstance(authorization, str)
                or not authorization.startswith("Bearer ")
            ):
                return JSONResponse(
                    status_code=401,
                    content={"error_code": "score_request_auth_required"},
                )
            token = authorization.removeprefix("Bearer ")
            token_sha256 = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if not hmac.compare_digest(
                token_sha256,
                authority.request_bearer_token_sha256,
            ):
                return JSONResponse(
                    status_code=403,
                    content={"error_code": "score_request_auth_invalid"},
                )
        if request.method == "POST" and request.url.path.startswith(JOBS_PATH):
            raw_length = request.headers.get("content-length")
            if raw_length is None or not raw_length.isdigit():
                return JSONResponse(
                    status_code=411,
                    content={"error_code": "score_content_length_required"},
                )
            maximum = (
                authority.max_request_body_bytes
                if request.url.path == JOBS_PATH
                else authority.max_action_body_bytes
            )
            if int(raw_length) > maximum:
                return JSONResponse(
                    status_code=413,
                    content={"error_code": "score_request_body_too_large"},
                )
        return await call_next(request)

    def call(operation: Any) -> Any:
        try:
            return operation()
        except ScoreServiceError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"error_code": exc.code},
            ) from exc

    @app.get("/health")
    def health() -> dict[str, Any] | JSONResponse:
        payload = manager.health()
        if payload["status"] != "healthy":
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.get("/info")
    def info() -> dict[str, Any]:
        return manager.info()

    @app.post(JOBS_PATH, status_code=202)
    def submit(request: CandidateScoreRequest) -> dict[str, Any]:
        return response_payload(call(lambda: manager.submit(request)))

    @app.get(f"{JOBS_PATH}/{{job_id}}")
    def get(job_id: str) -> dict[str, Any]:
        return response_payload(call(lambda: manager.get(job_id)))

    @app.post(f"{JOBS_PATH}/{{job_id}}/cancel")
    def cancel(job_id: str, request: JobActionRequest) -> dict[str, Any]:
        if request.job_id != job_id:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "score_action_job_id_mismatch"},
            )
        return response_payload(call(lambda: manager.cancel(job_id)))

    @app.post(f"{JOBS_PATH}/{{job_id}}/cleanup")
    def cleanup(job_id: str, request: JobActionRequest) -> dict[str, Any]:
        if request.job_id != job_id:
            raise HTTPException(
                status_code=409,
                detail={"error_code": "score_action_job_id_mismatch"},
            )
        return response_payload(call(lambda: manager.cleanup(job_id)))

    @app.on_event("shutdown")
    def shutdown() -> None:
        manager.close()

    return app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the explicitly authorized Cloud Slot Craftax scorer"
    )
    parser.add_argument("--authority", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--log-level", required=True)
    args = parser.parse_args()
    authority_path = Path(args.authority)
    if not authority_path.is_absolute():
        raise SystemExit("--authority must be an absolute path")
    if not isinstance(args.host, str) or not args.host.strip():
        raise SystemExit("--host must be explicit")
    if isinstance(args.port, bool) or not 1 <= args.port <= 65535:
        raise SystemExit("--port must be from 1 through 65535")
    if args.log_level not in {"critical", "error", "warning", "info"}:
        raise SystemExit("--log-level must be critical, error, warning, or info")
    authority = ScorerAuthority.load(authority_path)
    app = create_app(authority)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
