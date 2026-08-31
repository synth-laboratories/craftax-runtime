"""Strict wire contract for the isolated Craftax scorer service."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ENTRYPOINT = re.compile(r"^[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*$")
_IDENTIFIER_FIELDS = (
    "request_id",
    "idempotency_key",
    "project_id",
    "run_id",
    "work_product_id",
    "candidate_id",
    "execution_contract_version",
    "task_id",
    "suite_id",
    "policy_identity",
    "deployment_id",
    "claim_id",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def require_identifier(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    if not value or value != value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a bounded canonical string")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ValueError(f"{field} must not contain control characters")
    return value


def require_sha1(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 40-character Git SHA")
    return value


def require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class CandidateScoreIdentity(StrictModel):
    request_id: str
    idempotency_key: str
    project_id: str
    run_id: str
    work_product_id: str
    candidate_id: str
    candidate_sha256: str
    execution_contract_version: str
    entrypoint: str
    task_id: str
    suite_id: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    policy_identity: str
    gamebench_source_sha: str
    scorer_source_sha: str
    scorer_fixture_manifest_sha256: str
    scorer_binary_sha256: str
    scorer_image_digest: str
    network_mode: Literal["none"]
    environment_allowlist: tuple[str, ...]
    timeout_seconds: float
    environment: Literal["dev", "staging", "prod"]
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"]
    deployment_id: str
    claim_id: str
    fencing_token: int

    @field_validator("seeds", "environment_allowlist", mode="before")
    @classmethod
    def arrays_to_tuples(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_identity(self) -> CandidateScoreIdentity:
        for field in _IDENTIFIER_FIELDS:
            require_identifier(getattr(self, field), field)
        require_sha256(self.candidate_sha256, "candidate_sha256")
        require_sha1(self.gamebench_source_sha, "gamebench_source_sha")
        require_sha1(self.scorer_source_sha, "scorer_source_sha")
        require_sha256(
            self.scorer_fixture_manifest_sha256,
            "scorer_fixture_manifest_sha256",
        )
        require_sha256(self.scorer_binary_sha256, "scorer_binary_sha256")
        if _IMAGE_DIGEST.fullmatch(self.scorer_image_digest) is None:
            raise ValueError("scorer_image_digest must be immutable")
        if (
            not self.entrypoint
            or self.entrypoint != self.entrypoint.strip()
            or _SAFE_ENTRYPOINT.fullmatch(self.entrypoint) is None
            or self.entrypoint.startswith("/")
            or ".." in self.entrypoint.split("/")
        ):
            raise ValueError("entrypoint must be a safe canonical relative path")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("seeds must be non-empty and unique")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or seed > 2_147_483_647
            for seed in self.seeds
        ):
            raise ValueError("seeds must be non-negative signed 32-bit integers")
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.environment_allowlist != ():
            raise ValueError("environment_allowlist must be explicitly empty")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        if isinstance(self.fencing_token, bool) or self.fencing_token <= 0:
            raise ValueError("fencing_token must be a positive integer")
        return self

    def identity_wire(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        payload["seeds"] = list(self.seeds)
        payload["environment_allowlist"] = []
        return payload


class CandidateScoreRequest(CandidateScoreIdentity):
    candidate_bytes: str

    @model_validator(mode="after")
    def validate_candidate(self) -> CandidateScoreRequest:
        try:
            decoded = base64.b64decode(self.candidate_bytes, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("candidate_bytes must be canonical base64") from exc
        if base64.b64encode(decoded).decode("ascii") != self.candidate_bytes:
            raise ValueError("candidate_bytes must be canonical base64")
        if not decoded or b"\x00" in decoded:
            raise ValueError("candidate_bytes must be non-empty and contain no NUL")
        try:
            decoded.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("candidate_bytes must be strict UTF-8") from exc
        if hashlib.sha256(decoded).hexdigest() != self.candidate_sha256:
            raise ValueError("candidate_bytes do not match candidate_sha256")
        return self

    def decoded_candidate_bytes(self) -> bytes:
        return base64.b64decode(self.candidate_bytes, validate=True)

    def authority_wire(self) -> dict[str, Any]:
        payload = self.identity_wire()
        payload.pop("candidate_bytes", None)
        return payload

    def request_body_sha256(self) -> str:
        return canonical_json_sha256(self.identity_wire())


class ScoreInputAuthority(StrictModel):
    schema_version: Literal["gamebench.craftax.score_input_authority.v1"]
    request_body_sha256: str
    candidate_sha256: str
    candidate_size_bytes: int
    idempotency_key: str

    @model_validator(mode="after")
    def validate_authority(self) -> ScoreInputAuthority:
        require_sha256(self.request_body_sha256, "request_body_sha256")
        require_sha256(self.candidate_sha256, "candidate_sha256")
        require_identifier(self.idempotency_key, "idempotency_key")
        if isinstance(self.candidate_size_bytes, bool) or self.candidate_size_bytes <= 0:
            raise ValueError("candidate_size_bytes must be positive")
        return self


class CandidateScoreSubmission(CandidateScoreIdentity):
    schema_version: Literal["gamebench.craftax.score_submission.v1"]
    job_id: str
    status: Literal["accepted"]
    created_at: str
    input_authority: ScoreInputAuthority


class ScoreRow(StrictModel):
    seed: int
    reward: float
    achievement_count: int
    achievements: tuple[str, ...]
    rollout_id: str
    policy_isolation: Literal["os_sandbox_observation_action.v2"]
    episode_status: Literal["succeeded"]

    @field_validator("achievements", mode="before")
    @classmethod
    def achievements_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_row(self) -> ScoreRow:
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("score row seed must be non-negative")
        if not math.isfinite(self.reward):
            raise ValueError("score row reward must be finite")
        if isinstance(self.achievement_count, bool) or self.achievement_count < 0:
            raise ValueError("achievement_count must be non-negative")
        if self.achievement_count != len(self.achievements):
            raise ValueError("achievement_count does not match achievements")
        if tuple(sorted(set(self.achievements))) != self.achievements:
            raise ValueError("achievements must be sorted and unique")
        require_identifier(self.rollout_id, "rollout_id")
        return self


class ScoreOutputAuthority(StrictModel):
    schema_version: Literal["gamebench.craftax.score_output_authority.v1"]
    score_rows_sha256: str
    candidate_sha256: str
    gamebench_source_sha: str
    scorer_source_sha: str
    scorer_fixture_manifest_sha256: str
    scorer_binary_sha256: str
    scorer_image_digest: str
    task_id: str
    suite_id: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    policy_identity: str
    isolation_contract: Literal["os_sandbox_observation_action.v2"]
    network_mode: Literal["none"]
    environment_allowlist: tuple[str, ...]
    candidate_suite_visible: Literal[False]
    candidate_evaluator_visible: Literal[False]
    terminal: Literal[True]
    environment: Literal["dev", "staging", "prod"]
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"]
    deployment_id: str
    claim_id: str
    fencing_token: int

    @field_validator("seeds", "environment_allowlist", mode="before")
    @classmethod
    def arrays_to_tuples(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value


class ScoreFailure(StrictModel):
    code: str
    origin: Literal["candidate", "trusted_supervisor", "trusted_scorer"]
    retryable: bool


class CandidateScoreResult(CandidateScoreIdentity):
    schema_version: Literal["gamebench.craftax.score_result.v1"]
    job_id: str
    result_id: str
    status: Literal["succeeded", "failed", "timed_out", "cancelled"]
    mean_reward: float | None
    benchmark_score: float | None
    score_rows: tuple[ScoreRow, ...]
    output_authority: ScoreOutputAuthority | None
    failure: ScoreFailure | None
    created_at: str
    started_at: str | None
    finished_at: str

    @field_validator("score_rows", mode="before")
    @classmethod
    def rows_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value


class CandidateScoreCancellation(CandidateScoreIdentity):
    schema_version: Literal["gamebench.craftax.score_cancellation.v1"]
    job_id: str
    result_id: str | None
    status: Literal["cancelled", "already_terminal"]
    process_group_terminated: bool
    cleanup_required: bool
    cancelled_at: str


class CandidateScoreCleanupReceipt(CandidateScoreIdentity):
    schema_version: Literal["gamebench.craftax.score_cleanup.v1"]
    job_id: str
    result_id: str | None
    status: Literal["cleaned"]
    workspace_removed: bool
    process_group_terminated: bool
    policy_sandboxes_removed: bool
    processes_remaining: int
    containers_remaining: int
    resources_remaining: int
    state_record_disposition: Literal["retained_until_external_archive"]
    cleaned_at: str


class JobActionRequest(StrictModel):
    job_id: str

    @model_validator(mode="after")
    def validate_job_id(self) -> JobActionRequest:
        require_identifier(self.job_id, "job_id")
        return self


def identity_payload(request: CandidateScoreRequest) -> dict[str, Any]:
    return request.authority_wire()


def response_payload(model: StrictModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


__all__ = [
    "CandidateScoreCancellation",
    "CandidateScoreCleanupReceipt",
    "CandidateScoreIdentity",
    "CandidateScoreRequest",
    "CandidateScoreResult",
    "CandidateScoreSubmission",
    "JobActionRequest",
    "ScoreFailure",
    "ScoreInputAuthority",
    "ScoreOutputAuthority",
    "ScoreRow",
    "canonical_json_sha256",
    "identity_payload",
    "require_identifier",
    "require_sha1",
    "require_sha256",
    "response_payload",
]
