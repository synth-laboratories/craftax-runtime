"""Monty policy HTTP service. GEPA does not speak this hostname."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException

from containers.codepolicy.monty import MontyLoadError, normalize_spec, resolve_policy
from containers.codepolicy.policy_subprocess import CandidatePolicyFailure
from containers.gepa_codepolicy.apply import sha256_text
from containers.gepa_codepolicy.process_policy import ProcessPolicy
from containers.gepa_codepolicy.rust_env import ENV_PROTOCOL, RustEnv, load_task
from containers.gepa_codepolicy.score import episode_reward, universe_size

TASK_ROOT = Path(__file__).resolve().parents[2]
SEED_PATH = TASK_ROOT / "policies" / "heuristic_baseline.py"
TASK_TEMPLATE = "tasks/policy_dev_template.json"


def _candidate_path() -> Path:
    root = Path(os.environ.get("CRAFTAX_GEPA_RUNTIME") or tempfile.mkdtemp(prefix="craftax-gepa-"))
    root.mkdir(parents=True, exist_ok=True)
    return root / "candidate.py"


def _load_child(source: str, path: Path) -> tuple[Any, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    spec = {"kind": "monty_python", "module": path.stem, "entry": "choose_actions", "path": str(path)}
    normalize_spec(spec, path=path)
    mode = os.environ.get("GAMEBENCH_MONTY_SANDBOX", "auto").strip().lower()
    if mode in {"process", "process_fallback"}:
        try:
            child = ProcessPolicy(path)
        except CandidatePolicyFailure as exc:
            raise HTTPException(
                status_code=422, detail={"compile_ok": False, "error": "candidate_policy_failure"}
            ) from exc
        receipt = dict(child.isolation_receipt)
        receipt["fallback"] = "process_policy"
        return child, receipt
    try:
        child = resolve_policy(spec, path=path)
        receipt = dict(getattr(child, "isolation_receipt", {}) or {})
        return child, receipt
    except CandidatePolicyFailure as exc:
        raise HTTPException(status_code=422, detail={"compile_ok": False, "error": "candidate_policy_failure"}) from exc
    except (MontyLoadError, RuntimeError):
        child = ProcessPolicy(path)
        receipt = dict(child.isolation_receipt)
        receipt["fallback"] = "process_policy"
        return child, receipt


def create_app(*, env_url: str, candidate_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="craftax-monty-policy")
    env_url = env_url.rstrip("/")
    if candidate_path is None:
        candidate_path = _candidate_path()
    state: dict[str, Any] = {
        "child": None,
        "source": SEED_PATH.read_text(encoding="utf-8") if SEED_PATH.is_file() else "",
        "receipt": {},
        "spec": {},
        "path": candidate_path,
    }

    def _replace(source: str) -> dict[str, Any]:
        previous = state.get("child")
        if previous is not None and hasattr(previous, "close"):
            try:
                previous.close()
            except Exception:  # noqa: BLE001
                pass
            state["child"] = None
        child, receipt = _load_child(source, state["path"])
        state["child"] = child
        state["source"] = source
        state["receipt"] = receipt
        state["spec"] = {
            "kind": "monty_python",
            "module": state["path"].stem,
            "entry": "choose_actions",
            "path": str(state["path"]),
        }
        return {
            "ok": True,
            "compile_ok": True,
            "content_hash": sha256_text(source),
            "pid": os.getpid(),
            "child_pid": receipt.get("child_pid") or receipt.get("policy_uid"),
            "isolation": receipt,
            "spec": state["spec"],
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        env = RustEnv(env_url).health()
        return {
            "status": "ok" if state["child"] is not None else "empty",
            "policy_kind": "code_policy.v1",
            "load": "monty_python",
            "pid": os.getpid(),
            "content_hash": sha256_text(state["source"]) if state["source"] else None,
            "env": env,
            "env_protocol": ENV_PROTOCOL,
        }

    @app.get("/metadata")
    def metadata() -> dict[str, Any]:
        return {
            "policy_kind": "code_policy.v1",
            "reload": "in_process",
            "load": "monty_python",
            "lever_ids": ["policy_script"],
            "env_protocol": ENV_PROTOCOL,
            "entrypoint": "choose_actions",
            "isolation": "isolated_policy_process",
            "spec": state.get("spec") or {},
        }

    @app.post("/load")
    def load(payload: dict[str, Any]) -> dict[str, Any]:
        source = payload.get("source")
        if source is None and payload.get("path"):
            source = Path(str(payload["path"])).read_text(encoding="utf-8")
        if source is None:
            source = state["source"]
        try:
            return _replace(str(source))
        except HTTPException:
            raise
        except CandidatePolicyFailure as exc:
            raise HTTPException(
                status_code=422,
                detail={"ok": False, "compile_ok": False, "error": "candidate_policy_failure"},
            ) from exc

    @app.post("/act")
    def act(payload: dict[str, Any]) -> dict[str, Any]:
        child = state.get("child")
        if child is None:
            raise HTTPException(status_code=409, detail="no monty policy loaded")
        try:
            decision = child(
                observation_text=payload.get("observation_text") or "",
                session=payload.get("session") or {},
                valid_actions=payload.get("valid_actions") or [],
                engine=None,
                seed=payload.get("seed"),
                ply=int(payload.get("ply") or 0),
                readout=payload.get("readout") or {},
            )
        except CandidatePolicyFailure as exc:
            raise HTTPException(status_code=422, detail={"error": "candidate_policy_failure"}) from exc
        return decision

    @app.post("/episode")
    def episode(payload: dict[str, Any]) -> dict[str, Any]:
        child = state.get("child")
        if child is None:
            raise HTTPException(status_code=409, detail="no monty policy loaded")
        seed = int(payload.get("seed") or 0)
        max_steps = int(payload.get("max_steps") or 60)
        task_path = str(payload.get("task_template") or TASK_TEMPLATE)
        env = RustEnv(str(payload.get("env_url") or env_url))
        task = load_task(task_path, seed=seed, task_root=TASK_ROOT)
        root = env.create_rollout(task, seed)
        rollout_id = str(root["rollout_id"])
        readout = root["readout"]
        latest = root
        turns: list[dict[str, Any]] = []
        ply = 0
        session = {"rollout_id": rollout_id, "ply": 0, "lane": "rust"}
        stop_reason = None
        try:
            while not latest.get("terminated") and not latest.get("truncated") and ply < max_steps:
                session["ply"] = ply
                decision = child(
                    observation_text=readout.get("observation_text") or "",
                    session=session,
                    valid_actions=readout.get("valid_actions") or [],
                    engine=None,
                    seed=seed,
                    ply=ply,
                    readout=readout,
                )
                if decision.get("stop_episode") is True:
                    stop_reason = str(decision.get("stop_reason") or "policy requested stop")
                    break
                actions = decision.get("actions") or ["noop"]
                action = str(actions[0] if actions else "noop")
                latest = env.step(rollout_id, action)
                readout = latest.get("readout") or {}
                private = (readout.get("private") or {})
                turns.append(
                    {
                        "ply": ply,
                        "action": action,
                        "policy_reason": decision.get("policy_reason") or "",
                        "achievements": sorted(private.get("achievements") or []),
                    }
                )
                ply += 1
            private = (latest.get("readout") or {}).get("private") or {}
            achievements = sorted(private.get("achievements") or [])
            events = []
            try:
                log = env.event_log(rollout_id)
                events = list(log.get("legacy") or log.get("events") or [])[:80]
            except Exception:  # noqa: BLE001
                events = []
            reward = episode_reward(achievements)
            return {
                "reward": reward,
                "achievements": achievements,
                "ticks": ply,
                "stop_reason": stop_reason,
                "terminated": bool(latest.get("terminated")),
                "truncated": bool(latest.get("truncated")),
                "universe_size": universe_size(),
                "rollout_id": rollout_id,
                "turns": turns,
                "events": events,
                "architecture": "monty_python_rust",
                "llm_calls": 0,
            }
        finally:
            env.delete(rollout_id)
            env.close()

    @app.post("/shutdown")
    def shutdown() -> dict[str, Any]:
        child = state.get("child")
        if child is not None and hasattr(child, "close"):
            child.close()
        state["child"] = None
        return {"ok": True}

    @app.on_event("startup")
    def startup() -> None:
        if state["source"]:
            _replace(state["source"])

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19310)
    parser.add_argument("--env-url", default=os.environ.get("GAMEBENCH_CRAFTAX_RUST_URL", "http://127.0.0.1:8098"))
    parser.add_argument("--candidate", default="")
    args = parser.parse_args()
    candidate = Path(args.candidate) if args.candidate else _candidate_path()
    app = create_app(env_url=args.env_url, candidate_path=candidate)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
