"""Craftax adapter for the shared GameBench policy sandbox."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SANDBOX_OWNER = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "codepolicy"
    / "policy_subprocess.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "gamebench_policy_subprocess_authority", _SANDBOX_OWNER
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load GameBench policy sandbox authority: {_SANDBOX_OWNER}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

CandidatePolicyFailure = _MODULE.CandidatePolicyFailure
IsolatedPolicyProcess = _MODULE.IsolatedPolicyProcess
start_isolated_policy = _MODULE.start_isolated_policy
POLICY_PROCESS_CANDIDATE_FAILURE_EXIT_CODE = (
    _MODULE.POLICY_PROCESS_CANDIDATE_FAILURE_EXIT_CODE
)
cleanup_isolated_policy_containers = _MODULE.cleanup_isolated_policy_containers

__all__ = [
    "CandidatePolicyFailure",
    "IsolatedPolicyProcess",
    "start_isolated_policy",
    "POLICY_PROCESS_CANDIDATE_FAILURE_EXIT_CODE",
    "cleanup_isolated_policy_containers",
]
