"""HTTP client for GameBench Craftax gold_rust."""

from __future__ import annotations

import json
from typing import Any

import httpx

ENV_PROTOCOL = "gamebench.craftax_gold_rust.v1"


class RustEnv:
    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any]:
        return self._client.get("/health").json()

    def create_rollout(self, task: dict[str, Any], seed: int) -> dict[str, Any]:
        response = self._client.post("/rollouts", json={"task": task, "seed": seed})
        response.raise_for_status()
        return response.json()

    def step(self, rollout_id: str, action: str) -> dict[str, Any]:
        response = self._client.post(f"/rollouts/{rollout_id}/step", json={"action": action})
        response.raise_for_status()
        return response.json()

    def event_log(self, rollout_id: str) -> dict[str, Any]:
        response = self._client.get(f"/rollouts/{rollout_id}/event_log")
        response.raise_for_status()
        return response.json()

    def delete(self, rollout_id: str) -> None:
        try:
            self._client.delete(f"/rollouts/{rollout_id}")
        except httpx.HTTPError:
            return


def load_task(task_path: str, *, seed: int, task_root) -> dict[str, Any]:
    task = json.loads((task_root / task_path).read_text(encoding="utf-8"))
    task["seed"] = seed
    task["task_id"] = f"{task.get('task_id', 'craftax_policy_dev')}_{seed}"
    return task
