"""Strict backend read boundary for active CloudDeployment claim authority."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import model_validator

from containers.scorer_service.contract import (
    StrictModel,
    require_identifier,
    require_sha1,
    require_sha256,
)


class ClaimAuthorityError(RuntimeError):
    """The backend did not prove the exact active fenced claim."""


class TopologySource(StrictModel):
    schema_version: Literal["cloud-deployment-topology-source-v1"]
    kind: Literal["backend_git"]
    repository: Literal["https://github.com/synth-laboratories/backend.git"]
    implementation_commit_sha: str
    topology_id: str
    topology_version: str
    topology_digest_sha256: str

    @model_validator(mode="after")
    def validate_source(self) -> TopologySource:
        require_sha1(self.implementation_commit_sha, "implementation_commit_sha")
        require_identifier(self.topology_id, "topology_id")
        require_identifier(self.topology_version, "topology_version")
        require_sha256(self.topology_digest_sha256, "topology_digest_sha256")
        return self


class ActiveClaim(StrictModel):
    claim_id: str
    deployment_id: str
    org_id: str
    holder: str
    purpose: str
    ttl_seconds: int
    fencing_token: int
    state: str
    acquired_at: str | None
    expires_at: str | None
    released_at: str | None
    created_at: str | None
    updated_at: str | None
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"] | None
    topology_id: str
    topology_version: str
    topology_source: TopologySource | None
    vm_name: str | None

    @model_validator(mode="after")
    def validate_claim(self) -> ActiveClaim:
        for field in (
            "claim_id",
            "deployment_id",
            "org_id",
            "holder",
            "purpose",
            "topology_id",
            "topology_version",
        ):
            require_identifier(getattr(self, field), field)
        if isinstance(self.fencing_token, bool) or self.fencing_token <= 0:
            raise ValueError("active claim fencing_token must be positive")
        if isinstance(self.ttl_seconds, bool) or self.ttl_seconds <= 0:
            raise ValueError("active claim ttl_seconds must be positive")
        if self.acquired_at is None or self.expires_at is None:
            raise ValueError("active claim timestamps must be explicit")
        _parse_datetime(self.acquired_at, "acquired_at")
        _parse_datetime(self.expires_at, "expires_at")
        if self.released_at is not None:
            _parse_datetime(self.released_at, "released_at")
        if self.created_at is not None:
            _parse_datetime(self.created_at, "created_at")
        if self.updated_at is not None:
            _parse_datetime(self.updated_at, "updated_at")
        if self.topology_source is not None and (
            self.topology_source.topology_id != self.topology_id
            or self.topology_source.topology_version != self.topology_version
        ):
            raise ValueError("active claim topology source mismatch")
        return self

    def expires_at_datetime(self) -> datetime:
        if self.expires_at is None:
            raise ValueError("expires_at is required for an active claim")
        return _parse_datetime(self.expires_at, "expires_at")


class ClaimProjection(StrictModel):
    deployment_id: str
    cloud_slot: Literal["slot1-cloud", "slot2-cloud"] | None
    topology_id: str
    topology_version: str
    topology_source: TopologySource | None
    vm_name: str | None
    active_claim: ActiveClaim | None
    last_fencing_token: int | None

    @model_validator(mode="after")
    def validate_projection(self) -> ClaimProjection:
        require_identifier(self.deployment_id, "deployment_id")
        require_identifier(self.topology_id, "topology_id")
        require_identifier(self.topology_version, "topology_version")
        if self.topology_source is not None and (
            self.topology_source.topology_id != self.topology_id
            or self.topology_source.topology_version != self.topology_version
        ):
            raise ValueError("claim projection topology source mismatch")
        if self.active_claim is not None:
            expected = {
                "deployment_id": self.deployment_id,
                "cloud_slot": self.cloud_slot,
                "topology_id": self.topology_id,
                "topology_version": self.topology_version,
                "topology_source": self.topology_source,
                "vm_name": self.vm_name,
            }
            for field, value in expected.items():
                if getattr(self.active_claim, field) != value:
                    raise ValueError(f"active claim projection mismatch for {field}")
            if self.active_claim.fencing_token != self.last_fencing_token:
                raise ValueError("active claim does not match last_fencing_token")
        return self


def _parse_datetime(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be an explicit RFC3339 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be RFC3339") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


class ClaimAuthorityClient:
    """Read and validate current claim state without passing auth to workers."""

    def __init__(
        self,
        *,
        backend_api_base_url: str,
        service_auth_file: Path,
        timeout_seconds: float,
        environment: str,
        cloud_slot: str,
        deployment_id: str,
        claim_id: str,
        fencing_token: int,
    ) -> None:
        parsed = urlsplit(backend_api_base_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or backend_api_base_url.endswith("/")
        ):
            raise ValueError(
                "backend_api_base_url must be an explicit canonical HTTPS origin"
            )
        if not service_auth_file.is_absolute() or not service_auth_file.is_file():
            raise ValueError("service_auth_file must be an existing absolute file")
        if service_auth_file.stat().st_mode & 0o077:
            raise ValueError("service_auth_file must not be group/world accessible")
        if timeout_seconds <= 0:
            raise ValueError("claim read timeout must be positive")
        if environment not in {"dev", "staging", "prod"}:
            raise ValueError("environment is not a remote Synth environment")
        if cloud_slot not in {"slot1-cloud", "slot2-cloud"}:
            raise ValueError("cloud_slot is not canonical")
        require_identifier(deployment_id, "deployment_id")
        require_identifier(claim_id, "claim_id")
        if isinstance(fencing_token, bool) or fencing_token <= 0:
            raise ValueError("fencing_token must be positive")
        self._backend_api_base_url = backend_api_base_url
        self._service_auth_file = service_auth_file
        self._timeout_seconds = timeout_seconds
        self._environment = environment
        self._cloud_slot = cloud_slot
        self._deployment_id = deployment_id
        self._claim_id = claim_id
        self._fencing_token = fencing_token

    def assert_current(self) -> ClaimProjection:
        token = self._read_service_token()
        url = (
            f"{self._backend_api_base_url}/smr/v1/deployments/"
            f"{self._deployment_id}/claims"
        )
        try:
            response = httpx.get(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "X-Synth-Environment": self._environment,
                },
                timeout=self._timeout_seconds,
                follow_redirects=False,
            )
        except httpx.HTTPError as exc:
            raise ClaimAuthorityError("cloud claim authority read failed") from exc
        if response.status_code != 200:
            raise ClaimAuthorityError("cloud claim authority read was not successful")
        try:
            payload = response.json()
            projection = ClaimProjection.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ClaimAuthorityError("cloud claim projection was not typed") from exc
        active = projection.active_claim
        if (
            projection.deployment_id != self._deployment_id
            or projection.cloud_slot != self._cloud_slot
            or active is None
            or active.deployment_id != self._deployment_id
            or active.cloud_slot != self._cloud_slot
            or active.claim_id != self._claim_id
            or active.fencing_token != self._fencing_token
            or projection.last_fencing_token != self._fencing_token
            or active.released_at is not None
            or active.state != "active"
            or active.expires_at_datetime() <= datetime.now(UTC)
        ):
            raise ClaimAuthorityError("cloud claim authority is stale or mismatched")
        return projection

    def _read_service_token(self) -> str:
        try:
            payload = json.loads(self._service_auth_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ClaimAuthorityError("service auth source is unreadable") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "schema_version",
            "bearer_token",
        }:
            raise ClaimAuthorityError("service auth source has an invalid schema")
        if payload["schema_version"] != "gamebench.scorer_service_auth.v1":
            raise ClaimAuthorityError("service auth source version is unsupported")
        token = payload["bearer_token"]
        if (
            not isinstance(token, str)
            or not token
            or token != token.strip()
            or any(ord(char) < 0x21 or ord(char) > 0x7E for char in token)
        ):
            raise ClaimAuthorityError("service auth token is invalid")
        return token


__all__ = ["ClaimAuthorityClient", "ClaimAuthorityError", "ClaimProjection"]
