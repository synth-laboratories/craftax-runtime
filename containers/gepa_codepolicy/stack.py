"""Start gold_rust + Monty policy service + GEPA orchestrator."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

TASK_ROOT = Path(__file__).resolve().parents[2]
RUST_DIR = TASK_ROOT / "gold_rust"
RUST_BIN = RUST_DIR / "target" / "release" / "craftax_gold"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_http(url: str, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    last: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=0.5)
            if response.status_code < 500:
                return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not become healthy: {last}")


def _stop_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=6)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def ensure_rust_binary() -> Path:
    if RUST_BIN.is_file():
        return RUST_BIN
    subprocess.run(  # noqa: S603
        ["cargo", "build", "--release", "--quiet", "--bin", "craftax_gold"],
        cwd=str(RUST_DIR),
        check=True,
    )
    if not RUST_BIN.is_file():
        raise RuntimeError(f"missing rust binary {RUST_BIN}")
    return RUST_BIN


def _pythonpath() -> str:
    parts = [str(TASK_ROOT), str(TASK_ROOT / "gold_python"), str(TASK_ROOT / "shared")]
    existing = os.environ.get("PYTHONPATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


@dataclass
class GepaStack:
    env_url: str
    policy_url: str
    orch_url: str
    env_proc: subprocess.Popen | None
    policy_proc: subprocess.Popen
    orch_proc: subprocess.Popen
    owns_env: bool

    def stop(self) -> None:
        _stop_proc(self.orch_proc)
        _stop_proc(self.policy_proc)
        if self.owns_env:
            _stop_proc(self.env_proc)


def start_stack(
    *,
    env_url: str | None = None,
    env_port: int | None = None,
    policy_port: int | None = None,
    orch_port: int | None = None,
) -> GepaStack:
    owns_env = False
    env_proc = None
    if env_url:
        orch_env = env_url.rstrip("/")
    else:
        port = env_port or int(os.environ.get("GAMEBENCH_CRAFTAX_RUST_PORT", "8098"))
        orch_env = f"http://127.0.0.1:{port}"
        try:
            httpx.get(f"{orch_env}/health", timeout=0.4)
        except Exception:  # noqa: BLE001
            owns_env = True
            binary = ensure_rust_binary()
            env_proc = subprocess.Popen(  # noqa: S603
                [str(binary), "--host", "127.0.0.1", "--port", str(port)],
                cwd=str(RUST_DIR),
            )
            _wait_http(f"{orch_env}/health", timeout=90.0)
    policy_port = policy_port or _free_port()
    orch_port = orch_port or _free_port()
    policy_url = f"http://127.0.0.1:{policy_port}"
    orch_url = f"http://127.0.0.1:{orch_port}"
    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath()
    env["GAMEBENCH_CRAFTAX_RUST_URL"] = orch_env
    env.setdefault("GAMEBENCH_MONTY_SANDBOX", "process")
    policy_proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "containers.gepa_codepolicy.policy_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(policy_port),
            "--env-url",
            orch_env,
        ],
        cwd=str(TASK_ROOT),
        env=env,
    )
    _wait_http(f"{policy_url}/health", timeout=30.0)
    orch_proc = subprocess.Popen(  # noqa: S603
        [
            sys.executable,
            "-m",
            "containers.gepa_codepolicy.gepa_app",
            "--host",
            "127.0.0.1",
            "--port",
            str(orch_port),
            "--env-url",
            orch_env,
            "--policy-url",
            policy_url,
        ],
        cwd=str(TASK_ROOT),
        env=env,
    )
    _wait_http(f"{orch_url}/health", timeout=15.0)
    return GepaStack(
        env_url=orch_env,
        policy_url=policy_url,
        orch_url=orch_url,
        env_proc=env_proc,
        policy_proc=policy_proc,
        orch_proc=orch_proc,
        owns_env=owns_env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GEPA Monty Python on Craftax gold_rust")
    parser.add_argument("--env-url", default=os.environ.get("GAMEBENCH_CRAFTAX_RUST_URL"))
    parser.add_argument("--env-port", type=int, default=8098)
    parser.add_argument("--policy-port", type=int, default=19310)
    parser.add_argument("--orch-port", type=int, default=19300)
    args = parser.parse_args()
    stack = start_stack(
        env_url=args.env_url or None,
        env_port=args.env_port,
        policy_port=args.policy_port,
        orch_port=args.orch_port,
    )
    print(f"env     {stack.env_url}")
    print(f"policy  {stack.policy_url}")
    print(f"gepa    {stack.orch_url}")
    try:
        stack.orch_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stack.stop()


if __name__ == "__main__":
    main()
