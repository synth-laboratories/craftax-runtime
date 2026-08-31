#!/usr/bin/env python3
"""Start or stop one claim-bound Craftax scorer on a Synth Cloud Slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


CONTAINER_NAME = "craftax-scorer"
SERVICE_AUTH_PATH = "/run/gamebench/service-auth.json"
AUTHORITY_PATH = "/run/gamebench/scorer-authority.json"


def _run(argv: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=capture,
        text=True,
        timeout=120,
    )


def _required_sha(value: str, length: int, field: str) -> str:
    if re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise RuntimeError(f"{field} must be a lowercase {length}-character hash")
    return value


def _required_digest(value: str, field: str) -> str:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{field} must be an immutable sha256 digest")
    return value


def _required_local_image_ref(value: str) -> str:
    if (
        value != value.strip()
        or not value
        or any(character.isspace() for character in value)
        or value.startswith("sha256:")
    ):
        raise RuntimeError("local_image_ref must be an exact named local image reference")
    return value


def _loaded_image_identity(args: argparse.Namespace) -> tuple[str, str]:
    image_ref = _required_local_image_ref(args.local_image_ref)
    expected_manifest_digest = _required_digest(
        args.image_manifest_digest,
        "image_manifest_digest",
    )
    _required_digest(
        args.image_config_digest,
        "image_config_digest",
    )
    inspected = _run(["docker", "image", "inspect", image_ref])
    if inspected.returncode != 0:
        raise RuntimeError("Synth-materialized Craftax scorer image is absent locally")
    try:
        payload = json.loads(inspected.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("local Craftax scorer image inspection is malformed") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("local Craftax scorer image inspection is ambiguous")
    image = payload[0]
    image_id = image.get("Id")
    if image_id != expected_manifest_digest:
        raise RuntimeError("local Craftax scorer image manifest digest mismatch")
    repo_tags = image.get("RepoTags")
    repo_digests = image.get("RepoDigests")
    exact_refs = {
        value
        for values in (repo_tags, repo_digests)
        if isinstance(values, list)
        for value in values
        if isinstance(value, str)
    }
    if image_ref not in exact_refs:
        raise RuntimeError("local Craftax scorer image reference is not bound to inspected image")
    if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
        raise RuntimeError("local Craftax scorer image platform must be linux/amd64")
    return image_ref, expected_manifest_digest


def _slot_secret(path: Path, name: str) -> str:
    if not path.is_file() or path.stat().st_mode & 0o077:
        raise RuntimeError("slot compose env must be an existing private file")
    matches = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            matches.append(line.split("=", 1)[1])
    if len(matches) != 1 or not matches[0] or matches[0] != matches[0].strip():
        raise RuntimeError(f"slot compose env has no exact {name} authority")
    return matches[0]


def _write_private_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _identity_labels(args: argparse.Namespace) -> dict[str, str]:
    return {
        "ai.synth.cloud-deployment-id": args.deployment_id,
        "ai.synth.cloud-claim-id": args.claim_id,
        "ai.synth.cloud-fencing-token-sha256": hashlib.sha256(
            str(args.fencing_token).encode()
        ).hexdigest(),
        "ai.synth.scorer-image-digest": args.image_manifest_digest,
        "ai.synth.scorer-image-config-digest": args.image_config_digest,
        "ai.synth.scorer-local-image-ref": args.local_image_ref,
    }


def _labels(args: argparse.Namespace) -> list[str]:
    return [
        part
        for key, value in _identity_labels(args).items()
        for part in ("--label", f"{key}={value}")
    ]


def _assert_launched_identity(
    args: argparse.Namespace,
    image_manifest_digest: str,
) -> None:
    inspected = _run(["docker", "container", "inspect", CONTAINER_NAME])
    if inspected.returncode != 0:
        raise RuntimeError("launched Craftax scorer container inspection failed")
    try:
        payload = json.loads(inspected.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("launched Craftax scorer container inspection is malformed") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeError("launched Craftax scorer container inspection is ambiguous")
    container = payload[0]
    config = container.get("Config")
    labels = config.get("Labels") if isinstance(config, dict) else None
    if container.get("Image") != image_manifest_digest or not isinstance(labels, dict):
        raise RuntimeError("launched Craftax scorer image identity mismatch")
    if any(labels.get(key) != value for key, value in _identity_labels(args).items()):
        raise RuntimeError("launched Craftax scorer authority labels mismatch")


def start(args: argparse.Namespace) -> dict[str, Any]:
    if _run(["docker", "container", "inspect", CONTAINER_NAME]).returncode == 0:
        raise RuntimeError("craftax scorer container already exists")
    image_ref, image_manifest_digest = _loaded_image_identity(args)
    image_manifest_digest = _required_digest(
        args.image_manifest_digest,
        "image_manifest_digest",
    )
    token = _slot_secret(Path(args.slot_env_file), "SYNTH_API_KEY")
    root = Path(args.state_root).resolve()
    state = root / "state"
    workspace = root / "workspaces"
    auth = root / "auth"
    for directory in (root, state, workspace, auth):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    service_auth = auth / "service-auth.json"
    authority = auth / "scorer-authority.json"
    _write_private_json(
        service_auth,
        {
            "schema_version": "gamebench.scorer_service_auth.v1",
            "bearer_token": token,
        },
    )
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or len(seeds) != len(set(seeds)):
        raise RuntimeError("seeds must be a non-empty unique integer list")
    scorer_authority = {
        "schema_version": "gamebench.craftax.scorer_authority.v1",
        "environment": args.environment,
        "cloud_slot": args.cloud_slot,
        "deployment_id": args.deployment_id,
        "claim_id": args.claim_id,
        "fencing_token": args.fencing_token,
        "gamebench_source_sha": _required_sha(args.gamebench_source_sha, 40, "gamebench_source_sha"),
        "scorer_source_sha": _required_sha(args.scorer_source_sha, 40, "scorer_source_sha"),
        "scorer_fixture_manifest_sha256": _required_sha(args.fixture_manifest_sha256, 64, "fixture_manifest_sha256"),
        "scorer_binary_sha256": _required_sha(args.scorer_binary_sha256, 64, "scorer_binary_sha256"),
        "scorer_image_digest": image_manifest_digest,
        "backend_api_base_url": args.backend_api_base_url.rstrip("/"),
        "service_auth_file": SERVICE_AUTH_PATH,
        "request_bearer_token_sha256": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "backend_claim_read_timeout_seconds": 10.0,
        "expected_platform_system": "Linux",
        "expected_platform_machine": "x86_64",
        "state_directory": "/var/lib/gamebench/scorer-state",
        "workspace_directory": "/var/lib/gamebench/scorer-workspaces",
        "max_candidate_bytes": 65_536,
        "max_request_body_bytes": 200_000,
        "max_action_body_bytes": 4_096,
        "max_active_jobs": 1,
        "max_queued_jobs": 8,
        "max_retained_jobs": 16,
        "max_cleaned_state_records": 128,
        "cleaned_state_record_policy": "retain_until_external_archive",
        "maximum_timeout_seconds": 2_400.0,
        "process_termination_grace_seconds": 10.0,
        "episode_parallelism": 1,
        "profiles": [
            {
                "execution_contract_version": "gamebench.code_policy.v1",
                "entrypoint": "heuristic_policy.py",
                "task_id": "craftax-singleplayer",
                "suite_id": "rungamebench_craftax_heldout_v1",
                "seeds": list(seeds),
                "max_steps": args.max_steps,
                "lane": "rust",
                "policy_identity": "git_source_sha256_v1",
                "task_template": "tasks/policy_batch_template.json",
            }
        ],
    }
    _write_private_json(authority, scorer_authority)
    command = [
        "docker", "run", "--detach", "--name", CONTAINER_NAME,
        "--pull", "never",
        "--network", args.docker_network,
        "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--security-opt", "seccomp=unconfined",
        "--tmpfs", "/tmp:mode=0700,uid=65532,gid=65532",
        "--volume", f"{authority}:{AUTHORITY_PATH}:ro",
        "--volume", f"{service_auth}:{SERVICE_AUTH_PATH}:ro",
        "--volume", f"{state}:/var/lib/gamebench/scorer-state",
        "--volume", f"{workspace}:/var/lib/gamebench/scorer-workspaces",
        *_labels(args), image_manifest_digest,
        "--authority", AUTHORITY_PATH,
        "--host", "0.0.0.0", "--port", "8001", "--log-level", "info",
    ]
    launched = _run(command)
    if launched.returncode != 0:
        raise RuntimeError("Craftax scorer container launch failed")
    try:
        _assert_launched_identity(args, image_manifest_digest)
    except BaseException:
        _run(["docker", "rm", "--force", CONTAINER_NAME])
        raise
    try:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            health = _run(
                [
                    "docker", "exec", args.backend_container,
                    "python", "-c",
                    "import json,urllib.request; print(json.load(urllib.request.urlopen('http://craftax-scorer:8001/health',timeout=2))['status'])",
                ]
            )
            if health.returncode == 0 and health.stdout.strip() == "healthy":
                return {
                    "schema_version": "gamebench.craftax.cloud_scorer_launch.v1",
                    "status": "healthy",
                    "container_name": CONTAINER_NAME,
                    "deployment_id": args.deployment_id,
                    "claim_id": args.claim_id,
                    "fencing_token_sha256": hashlib.sha256(
                        str(args.fencing_token).encode()
                    ).hexdigest(),
                    "scorer_source_sha": args.scorer_source_sha,
                    "scorer_image_digest": image_manifest_digest,
                    "scorer_image_config_digest": args.image_config_digest,
                    "scorer_local_image_ref": image_ref,
                }
            time.sleep(1.0)
    except BaseException:
        _run(["docker", "rm", "--force", CONTAINER_NAME])
        raise
    _run(["docker", "rm", "--force", CONTAINER_NAME])
    raise RuntimeError("Craftax scorer did not become healthy")


def stop(args: argparse.Namespace) -> dict[str, Any]:
    inspect = _run(["docker", "container", "inspect", CONTAINER_NAME, "--format", "{{json .Config.Labels}}"])
    if inspect.returncode != 0:
        return {"status": "absent", "container_name": CONTAINER_NAME}
    labels = json.loads(inspect.stdout)
    expected = {
        "ai.synth.cloud-deployment-id": args.deployment_id,
        "ai.synth.cloud-claim-id": args.claim_id,
        "ai.synth.cloud-fencing-token-sha256": hashlib.sha256(
            str(args.fencing_token).encode()
        ).hexdigest(),
    }
    if any(labels.get(key) != value for key, value in expected.items()):
        raise RuntimeError("existing Craftax scorer identity does not match cleanup authority")
    removed = _run(["docker", "rm", "--force", CONTAINER_NAME])
    if removed.returncode != 0:
        raise RuntimeError("Craftax scorer cleanup failed")
    auth = Path(args.state_root).resolve() / "auth"
    for filename in ("service-auth.json", "scorer-authority.json"):
        (auth / filename).unlink(missing_ok=True)
    return {"status": "removed", "container_name": CONTAINER_NAME, **expected}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("start", "stop"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--deployment-id", required=True)
        sub.add_argument("--claim-id", required=True)
        sub.add_argument("--fencing-token", required=True, type=int)
        sub.add_argument("--state-root", required=True)
        if command == "start":
            sub.add_argument("--environment", required=True, choices=("dev", "staging", "prod"))
            sub.add_argument("--cloud-slot", required=True, choices=("slot1-cloud", "slot2-cloud"))
            sub.add_argument("--backend-api-base-url", required=True)
            sub.add_argument("--slot-env-file", required=True)
            sub.add_argument("--docker-network", required=True)
            sub.add_argument("--backend-container", required=True)
            sub.add_argument("--gamebench-source-sha", required=True)
            sub.add_argument("--scorer-source-sha", required=True)
            sub.add_argument("--fixture-manifest-sha256", required=True)
            sub.add_argument("--scorer-binary-sha256", required=True)
            sub.add_argument("--local-image-ref", required=True)
            sub.add_argument("--image-manifest-digest", required=True)
            sub.add_argument("--image-config-digest", required=True)
            sub.add_argument("--seeds", required=True)
            sub.add_argument("--max-steps", required=True, type=int)
    return result


def main() -> int:
    args = parser().parse_args()
    payload = start(args) if args.command == "start" else stop(args)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
