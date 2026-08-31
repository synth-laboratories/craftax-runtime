#!/usr/bin/env python3
"""Run a Craftax gold lane HTTP service."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import sys


TASK_DIR = Path(__file__).resolve().parents[1]
for path in (TASK_DIR, TASK_DIR / "gold_python", TASK_DIR / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["python", "rust"], default="python")
    parser.add_argument("--host", default=os.environ.get("GAMEBENCH_CRAFTAX_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Enable Craftax rust replay.gif / frame capture (sets GAMEBENCH_CRAFTAX_REPLAY_ENABLED=1)",
    )
    args = parser.parse_args()
    if args.lane == "python":
        port = args.port or int(os.environ.get("GAMEBENCH_CRAFTAX_PORT", "8097"))
        import shutil

        uvicorn_bin = shutil.which("uvicorn")
        if uvicorn_bin is None:
            raise SystemExit("uvicorn not found on PATH; install it for the Craftax python gold lane")
        subprocess.run(
            [
                uvicorn_bin,
                "service:app",
                "--host",
                args.host,
                "--port",
                str(port),
                "--app-dir",
                str(TASK_DIR / "gold_python"),
            ],
            check=True,
            cwd=str(TASK_DIR / "gold_python"),
            env={**os.environ, "PYTHONPATH": os.pathsep.join([str(TASK_DIR), str(TASK_DIR / "gold_python"), str(TASK_DIR / "shared")])},
        )
    else:
        port = args.port or int(os.environ.get("GAMEBENCH_CRAFTAX_RUST_PORT", "8098"))
        if args.replay:
            os.environ["GAMEBENCH_CRAFTAX_REPLAY_ENABLED"] = "1"
        subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "--quiet",
                "--manifest-path",
                str(TASK_DIR / "gold_rust" / "Cargo.toml"),
                "--bin",
                "craftax_gold",
                "--",
                "--host",
                args.host,
                "--port",
                str(port),
                *(["--replay"] if args.replay else []),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
