"""Explicit runtime and source authority for one Cloud Slot scorer deployment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator

from containers.codepolicy.rust_repl_session import (
    PREBUILT_BINARY,
    PREBUILT_MANIFEST,
    PrebuiltRustReplManifest,
    ensure_rust_repl_binary,
)
from containers.scorer_service.contract import (
    CandidateScoreRequest,
    StrictModel,
    require_identifier,
    require_sha1,
    require_sha256,
)

_MAX_REQUEST_METADATA_BYTES = 64 * 1024


class ScoringProfile(StrictModel):
    """One frozen task/suite/policy contract admitted by the scorer image."""

    execution_contract_version: str
    entrypoint: str
    task_id: str
    suite_id: str
    seeds: tuple[int, ...]
    max_steps: int
    lane: Literal["rust"]
    policy_identity: str
    task_template: str

    @field_validator("seeds", mode="before")
    @classmethod
    def seeds_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_profile(self) -> ScoringProfile:
        for field in (
            "execution_contract_version",
            "entrypoint",
            "task_id",
            "suite_id",
            "policy_identity",
            "task_template",
        ):
            require_identifier(getattr(self, field), field)
        if self.entrypoint.startswith("/") or ".." in self.entrypoint.split("/"):
            raise ValueError("profile entrypoint must be a safe relative path")
        if self.task_template.startswith("/") or ".." in self.task_template.split("/"):
            raise ValueError("profile task_template must be a safe relative path")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("profile seeds must be non-empty and unique")
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise ValueError("profile seeds must be non-negative integers")
        if isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ValueError("profile max_steps must be positive")
        return self

    def matches(self, request: CandidateScoreRequest) -> bool:
        policy_identity = re.fullmatch(
            r"git:([0-9a-f]{40}):([^:]+):sha256:([0-9a-f]{64})",
            request.policy_identity,
        )
        return (
            self.execution_contract_version == request.execution_contract_version
            and self.entrypoint == request.entrypoint
            and self.task_id == request.task_id
            and self.suite_id == request.suite_id
            and self.seeds == request.seeds
            and self.max_steps == request.max_steps
            and self.lane == request.lane
            and self.policy_identity == "git_source_sha256_v1"
            and policy_identity is not None
            and policy_identity.group(2) == request.entrypoint
            and policy_identity.group(3) == request.candidate_sha256
        )


class ScorerAuthority(StrictModel):
    """Build-injected authority; omission is a startup error, never a default."""

    schema_version: Literal["gamebench.craftax.scorer_authority.v1"]
    environment: Literal["dev", "staging", "prod"]
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"]
    deployment_id: str
    claim_id: str
    fencing_token: int
    gamebench_source_sha: str
    scorer_source_sha: str
    scorer_fixture_manifest_sha256: str
    scorer_binary_sha256: str
    scorer_image_digest: str
    backend_api_base_url: str
    service_auth_file: str
    request_bearer_token_sha256: str
    backend_claim_read_timeout_seconds: float
    expected_platform_system: Literal["Linux"]
    expected_platform_machine: Literal["x86_64"]
    state_directory: str
    workspace_directory: str
    max_candidate_bytes: int
    max_request_body_bytes: int
    max_action_body_bytes: int
    max_active_jobs: int
    max_queued_jobs: int
    max_retained_jobs: int
    max_cleaned_state_records: int
    cleaned_state_record_policy: Literal["retain_until_external_archive"]
    maximum_timeout_seconds: float
    process_termination_grace_seconds: float
    episode_parallelism: int
    profiles: tuple[ScoringProfile, ...]

    @field_validator("profiles", mode="before")
    @classmethod
    def profiles_to_tuple(cls, value: Any) -> Any:
        if isinstance(value, list):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_authority(self) -> ScorerAuthority:
        require_identifier(self.deployment_id, "deployment_id")
        require_identifier(self.claim_id, "claim_id")
        require_sha1(self.gamebench_source_sha, "gamebench_source_sha")
        require_sha1(self.scorer_source_sha, "scorer_source_sha")
        require_sha256(
            self.scorer_fixture_manifest_sha256,
            "scorer_fixture_manifest_sha256",
        )
        require_sha256(self.scorer_binary_sha256, "scorer_binary_sha256")
        if not self.scorer_image_digest.startswith("sha256:"):
            raise ValueError("scorer_image_digest must be an immutable sha256 digest")
        require_sha256(
            self.scorer_image_digest.removeprefix("sha256:"),
            "scorer_image_digest",
        )
        parsed_backend = urlsplit(self.backend_api_base_url)
        if (
            parsed_backend.scheme != "https"
            or not parsed_backend.hostname
            or parsed_backend.username is not None
            or parsed_backend.password is not None
            or parsed_backend.query
            or parsed_backend.fragment
            or parsed_backend.path not in {"", "/"}
            or self.backend_api_base_url.endswith("/")
        ):
            raise ValueError("backend_api_base_url must be a canonical HTTPS origin")
        auth_path = Path(self.service_auth_file)
        if not auth_path.is_absolute():
            raise ValueError("service_auth_file must be an absolute path")
        require_sha256(
            self.request_bearer_token_sha256,
            "request_bearer_token_sha256",
        )
        if isinstance(self.fencing_token, bool) or self.fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        for field in (
            "max_candidate_bytes",
            "max_request_body_bytes",
            "max_action_body_bytes",
            "max_active_jobs",
            "max_queued_jobs",
            "max_retained_jobs",
            "max_cleaned_state_records",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not self.profiles:
            raise ValueError("profiles must be non-empty")
        encoded_candidate_bytes = 4 * ((self.max_candidate_bytes + 2) // 3)
        largest_seed_vector_bytes = max(
            len(
                json.dumps(
                    list(profile.seeds),
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            for profile in self.profiles
        )
        minimum_request_body_bytes = (
            encoded_candidate_bytes
            + largest_seed_vector_bytes
            + _MAX_REQUEST_METADATA_BYTES
        )
        if self.max_request_body_bytes < minimum_request_body_bytes:
            raise ValueError(
                "max_request_body_bytes cannot carry the maximum canonical base64 "
                "candidate and bounded request metadata"
            )
        for field in (
            "maximum_timeout_seconds",
            "backend_claim_read_timeout_seconds",
        ):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and positive")
        if (
            not math.isfinite(self.process_termination_grace_seconds)
            or self.process_termination_grace_seconds < 5
        ):
            raise ValueError(
                "process_termination_grace_seconds must be at least 5 seconds"
            )
        if self.episode_parallelism != 1:
            raise ValueError(
                "episode_parallelism must be exactly 1 for process-group fencing"
            )
        profile_keys = {
            (profile.task_id, profile.suite_id, profile.policy_identity)
            for profile in self.profiles
        }
        if len(profile_keys) != len(self.profiles):
            raise ValueError("profiles contain an ambiguous authority key")
        state = Path(self.state_directory)
        workspace = Path(self.workspace_directory)
        if not state.is_absolute() or not workspace.is_absolute() or state == workspace:
            raise ValueError(
                "state_directory and workspace_directory must be distinct absolute paths"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> ScorerAuthority:
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError("scorer authority path must be an existing absolute file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("scorer authority file is not strict JSON") from exc
        return cls.model_validate(payload)

    def validate_runtime(self) -> None:
        """Fail before serving if platform or tracked scorer fixtures differ."""
        if platform.system() != self.expected_platform_system:
            raise RuntimeError(
                "scorer runtime platform mismatch: "
                f"expected={self.expected_platform_system}, actual={platform.system()}"
            )
        if platform.machine().lower() != self.expected_platform_machine:
            raise RuntimeError(
                "scorer runtime architecture mismatch: "
                f"expected={self.expected_platform_machine}, actual={platform.machine()}"
            )
        status_fields = {
            key.strip(): value.strip()
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
                if ":" in line
            )
        }
        if status_fields.get("Seccomp") != "0":
            raise RuntimeError(
                "scorer runtime must use the declared seccomp=unconfined security context"
            )
        bwrap = shutil.which("bwrap")
        if not bwrap:
            raise RuntimeError("scorer runtime requires bubblewrap")
        sandbox_probe_command = [
            bwrap,
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
        ]
        for system_path in (Path("/bin"), Path("/lib"), Path("/lib64")):
            if system_path.is_symlink():
                sandbox_probe_command.extend(
                    ("--symlink", os.readlink(system_path), str(system_path))
                )
            elif system_path.exists():
                sandbox_probe_command.extend(
                    ("--ro-bind", str(system_path), str(system_path))
                )
        sandbox_probe_command.extend(
            ("--proc", "/proc", "--dev", "/dev", "/usr/bin/true")
        )
        sandbox_probe = subprocess.run(
            sandbox_probe_command,
            check=False,
            capture_output=True,
            timeout=5,
        )
        if sandbox_probe.returncode != 0:
            raise RuntimeError("scorer runtime bubblewrap isolation preflight failed")
        manifest_bytes = PREBUILT_MANIFEST.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self.scorer_fixture_manifest_sha256:
            raise RuntimeError("scorer fixture manifest SHA mismatch")
        # Fixture provenance: the recorded aarch64 fixture must match its own
        # manifest. Execution authority is bound separately below against the
        # binary this host actually runs (x86_64 image build on cloud hosts),
        # so the fixture bytes are not compared to scorer_binary_sha256.
        manifest = PrebuiltRustReplManifest.load(PREBUILT_MANIFEST)
        manifest.verify_source_closure()
        manifest.verify_fixture_binary(PREBUILT_BINARY)
        materialized = ensure_rust_repl_binary()
        if hashlib.sha256(materialized.read_bytes()).hexdigest() != self.scorer_binary_sha256:
            raise RuntimeError("materialized scorer binary SHA mismatch")
        state = Path(self.state_directory)
        workspace = Path(self.workspace_directory)
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
        if state.stat().st_mode & 0o077 or workspace.stat().st_mode & 0o077:
            raise RuntimeError("scorer writable directories must not be group/world accessible")

    def assert_request(self, request: CandidateScoreRequest) -> ScoringProfile:
        """Require exact Cloud Slot, claim, source, image, and suite authority."""
        expected = {
            "environment": self.environment,
            "cloud_slot": self.cloud_slot,
            "deployment_id": self.deployment_id,
            "claim_id": self.claim_id,
            "fencing_token": self.fencing_token,
            "gamebench_source_sha": self.gamebench_source_sha,
            "scorer_source_sha": self.scorer_source_sha,
            "scorer_fixture_manifest_sha256": self.scorer_fixture_manifest_sha256,
            "scorer_binary_sha256": self.scorer_binary_sha256,
            "scorer_image_digest": self.scorer_image_digest,
        }
        for field, value in expected.items():
            if getattr(request, field) != value:
                raise ValueError(f"scorer request authority mismatch for {field}")
        candidate_size = len(request.decoded_candidate_bytes())
        if candidate_size > self.max_candidate_bytes:
            raise ValueError("candidate_bytes exceed the declared scorer authority limit")
        if request.timeout_seconds > self.maximum_timeout_seconds:
            raise ValueError("timeout_seconds exceed the declared scorer authority limit")
        matches = [profile for profile in self.profiles if profile.matches(request)]
        if len(matches) != 1:
            raise ValueError("score request has no unambiguous frozen profile")
        task_path = Path(__file__).resolve().parents[2] / matches[0].task_template
        if not task_path.is_file():
            raise RuntimeError("frozen profile task template is absent from scorer image")
        return matches[0]


__all__ = ["ScorerAuthority", "ScoringProfile"]
