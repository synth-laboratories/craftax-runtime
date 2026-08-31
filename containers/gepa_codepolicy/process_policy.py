"""JSONL policy child without Docker/bwrap. Same serve protocol as IsolatedPolicyProcess."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

from containers.codepolicy.policy_subprocess import (
    CandidatePolicyFailure,
    POLICY_PROCESS_CANDIDATE_FAILURE_EXIT_CODE,
)

_MAX_IPC_LINE_CHARS = 2 * 1024 * 1024
_SERVE = (
    Path(__file__).resolve().parents[3]
    / "shared"
    / "codepolicy"
    / "policy_subprocess.py"
)


class ProcessPolicy:
    """Subprocess `choose_actions` proxy. Candidate cannot see the suite tree."""

    def __init__(self, policy_path: Path) -> None:
        source_path = policy_path.expanduser().resolve()
        if not source_path.is_file():
            raise CandidatePolicyFailure()
        self._sandbox = tempfile.TemporaryDirectory(prefix="craftax-monty-process-")
        root = Path(self._sandbox.name)
        policy_copy = root / "policy.py"
        server_copy = root / "policy_subprocess.py"
        policy_copy.write_bytes(source_path.read_bytes())
        server_copy.write_bytes(_SERVE.read_bytes())
        self._proc = subprocess.Popen(  # noqa: S603
            [sys.executable, str(server_copy), "--serve", str(policy_copy)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=str(root),
            env={"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"},
            close_fds=True,
        )
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("process policy pipes unavailable")
        self._stdin = self._proc.stdin
        self._stdout = self._proc.stdout
        self._lock = threading.Lock()
        self._request_id = 0
        ready_line = self._stdout.readline()
        try:
            ready = json.loads(ready_line or "")
        except json.JSONDecodeError as exc:
            self.close()
            raise CandidatePolicyFailure() from exc
        if ready.get("op") != "ready" or ready.get("ok") is not True:
            self.close()
            raise CandidatePolicyFailure()
        self.isolation_receipt = {
            "contract": "process_observation_action.v1",
            "platform": sys.platform,
            "sandbox": "process_fallback",
            "network": "shared_with_evaluator",
            "filesystem": "process_cwd_policy_only",
            "suite_visible": False,
            "child_pid": self._proc.pid,
        }

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        session = dict(kwargs.get("session") or {})
        valid_actions = [str(action) for action in kwargs.get("valid_actions") or []]
        source_readout = dict(kwargs.get("readout") or {})
        public_readout = {
            key: source_readout[key]
            for key in ("ascii", "observation", "observation_text")
            if key in source_readout
        }
        public_readout["valid_actions"] = valid_actions
        request = {
            "op": "choose_actions",
            "observation_text": str(kwargs.get("observation_text") or ""),
            "session": {
                "lane": str(session.get("lane") or "rust"),
                "ply": int(session.get("ply") or 0),
            },
            "valid_actions": valid_actions,
            "ply": int(kwargs.get("ply") or 0),
            "readout": public_readout,
        }
        with self._lock:
            if self._proc.poll() is not None:
                raise CandidatePolicyFailure()
            self._request_id += 1
            request["id"] = self._request_id
            encoded = json.dumps(request, separators=(",", ":"))
            if len(encoded) > _MAX_IPC_LINE_CHARS:
                raise ValueError("isolated policy request exceeds IPC limit")
            try:
                self._stdin.write(encoded + "\n")
                self._stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise CandidatePolicyFailure() from exc
            response_line = self._stdout.readline()
        if not response_line:
            raise CandidatePolicyFailure()
        try:
            response = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise CandidatePolicyFailure() from exc
        if response.get("ok") is False:
            raise CandidatePolicyFailure()
        if response.get("id") != self._request_id or not isinstance(response.get("decision"), dict):
            raise CandidatePolicyFailure()
        return dict(response["decision"])

    def close(self) -> None:
        if self._proc.poll() is None:
            try:
                self._request_id += 1
                self._stdin.write(
                    json.dumps({"id": self._request_id, "op": "close"}, separators=(",", ":"))
                    + "\n"
                )
                self._stdin.flush()
                self._proc.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                if os.name == "posix":
                    try:
                        os.kill(self._proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                self._proc.kill()
        self._sandbox.cleanup()
