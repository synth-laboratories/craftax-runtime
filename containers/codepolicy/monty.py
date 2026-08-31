"""Monty policy load for Craftax: monty_python → IsolatedPolicyProcess."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from containers.codepolicy.policy_subprocess import IsolatedPolicyProcess

DEFAULT_ENTRY = "choose_actions"
DEFAULT_KIND = "monty_python"


class MontyLoadError(ValueError):
    """Invalid Monty spec or missing entrypoint."""


def normalize_spec(spec: dict[str, Any] | None, *, path: Path | None = None) -> dict[str, Any]:
    body = dict(spec or {})
    kind = str(body.get("kind") or DEFAULT_KIND)
    if kind != DEFAULT_KIND:
        raise MontyLoadError(f"unsupported monty kind: {kind}")
    entry = str(body.get("entry") or DEFAULT_ENTRY)
    if entry != DEFAULT_ENTRY:
        raise MontyLoadError(f"Craftax monty entry must be {DEFAULT_ENTRY}, got {entry}")
    resolved = Path(body.get("path") or path or "")
    if not resolved.is_file():
        raise MontyLoadError(f"policy file not found: {resolved}")
    module = str(body.get("module") or resolved.stem)
    return {
        "kind": DEFAULT_KIND,
        "module": module,
        "entry": entry,
        "path": str(resolved.expanduser().resolve()),
    }


def resolve_policy(spec: dict[str, Any] | None, *, path: Path | None = None) -> IsolatedPolicyProcess:
    """Load a Craftax `choose_actions` module behind observation/action JSONL IPC."""
    normalized = normalize_spec(spec, path=path)
    return IsolatedPolicyProcess(Path(normalized["path"]))
