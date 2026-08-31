"""Async HTTP client for Craftax rust gold."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urljoin

import httpx


class CraftaxGoldRequestError(RuntimeError):
    """Raised when the Craftax gold HTTP lane returns a non-success response."""


class CraftaxRustGoldClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (
            base_url or os.environ.get("CRAFTAX_GOLD_URL", "http://127.0.0.1:8103")
        ).rstrip("/")

    async def _json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 120.0,
    ) -> dict[str, Any]:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(method, url, json=payload)
        if response.is_error:
            body = response.text[:500]
            raise CraftaxGoldRequestError(
                f"craftax gold {method} {path} failed status={response.status_code} body={body}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise CraftaxGoldRequestError(
                f"craftax gold {method} {path} returned non-object JSON: {type(data).__name__}"
            )
        return data

    async def health(self) -> dict[str, Any]:
        payload = await self._json("GET", "/health", timeout=10.0)
        if not payload.get("ok"):
            raise CraftaxGoldRequestError(
                f"craftax gold /health reported unhealthy payload={payload}"
            )
        return payload

    async def create_rollout(
        self, *, task: dict[str, Any], seed: int
    ) -> dict[str, Any]:
        return await self._json("POST", "/rollouts", {"task": task, "seed": seed})

    async def readout(self, rollout_id: str) -> dict[str, Any]:
        return await self._json("GET", f"/rollouts/{rollout_id}/readout")

    async def step(self, rollout_id: str, action: str) -> dict[str, Any]:
        return await self._json(
            "POST", f"/rollouts/{rollout_id}/step", {"action": action}
        )

    async def checkpoint_with_blob(self, rollout_id: str) -> dict[str, Any]:
        payload = await self._json("POST", f"/rollouts/{rollout_id}/checkpoint", None)
        blob = str(payload.get("blob") or "").strip()
        if not blob:
            raise CraftaxGoldRequestError(
                f"craftax gold checkpoint for rollout_id={rollout_id} missing blob field"
            )
        return payload

    async def restore(self, rollout_id: str, blob_b64: str) -> dict[str, Any]:
        return await self._json(
            "POST", f"/rollouts/{rollout_id}/restore", {"blob": blob_b64}
        )
