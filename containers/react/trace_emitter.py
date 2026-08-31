"""First-class Containers Trace V5 emission for the Craftax ReAct workload.

The workload owns the facts it observes.  ``synth_containers`` owns their trace
schema, transport, normalization, and storage.  This module only translates the
rollout request's propagated trace context into the public remote-emitter API.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from synth_containers.tracing import TraceContextV1
from synth_containers.tracing.capture.emitter import TraceEmitter

EXPECTED_CONTAINERS_VERSION = "0.4.0.20260725"
if version("synth-containers") != EXPECTED_CONTAINERS_VERSION:
    raise RuntimeError(
        "Craftax Trace V5 requires the central synth-containers "
        f"{EXPECTED_CONTAINERS_VERSION} build-context package"
    )


class CraftaxTrace:
    """Optional emitter with strict behavior whenever capture was requested."""

    def __init__(
        self,
        agent_emitter: TraceEmitter | None,
        environment_emitter: TraceEmitter | None = None,
    ) -> None:
        self._agent_emitter = agent_emitter
        self._environment_emitter = environment_emitter

    @classmethod
    def from_request(cls, request: Mapping[str, Any]) -> "CraftaxTrace":
        raw = request.get("trace_context")
        if raw is None:
            return cls(None)
        if not isinstance(raw, Mapping):
            raise ValueError("trace_context must be a mapping of propagated environment values")
        environment: dict[str, str] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("trace_context keys and values must be strings")
            environment[key] = value
        context = TraceContextV1.from_environment(environment)
        if context is None or not context.collector_url:
            raise ValueError("trace_context must contain complete Synth identity and collector URL")
        collector_token = _required_context_value(raw, "SYNTH_TRACE_COLLECTOR_TOKEN")
        environment_collector_token = _required_context_value(
            raw,
            "SYNTH_ENVIRONMENT_COLLECTOR_TOKEN",
        )
        environment_capture_id = _required_context_value(
            raw,
            "SYNTH_ENVIRONMENT_CAPTURE_ID",
        )
        environment_actor_id = _required_context_value(raw, "SYNTH_ENVIRONMENT_ACTOR_ID")
        environment_session_id = _required_context_value(
            raw,
            "SYNTH_ENVIRONMENT_SESSION_ID",
        )
        environment_delegation_id = _required_context_value(
            raw,
            "SYNTH_ENVIRONMENT_DELEGATION_ID",
        )
        if environment_capture_id == context.capture_id:
            raise ValueError("environment capture must be distinct from the policy capture")
        if environment_actor_id == context.actor_id:
            raise ValueError("environment actor must be distinct from the policy actor")
        environment_context = TraceContextV1(
            trace_id=context.trace_id,
            capture_id=environment_capture_id,
            actor_id=environment_actor_id,
            actor_session_id=environment_session_id,
            parent_actor_id=context.actor_id,
            parent_actor_session_id=context.actor_session_id,
            delegation_id=environment_delegation_id,
            workflow_address="gamebench/craftax/environment",
            binding_path=context.binding_path,
            collector_url=context.collector_url,
            output_dir=context.output_dir,
        )
        return cls(
            TraceEmitter(
                base_url=context.collector_url,
                context=context,
                collector_token=collector_token,
            ),
            TraceEmitter(
                base_url=context.collector_url,
                context=environment_context,
                collector_token=environment_collector_token,
            ),
        )

    @property
    def enabled(self) -> bool:
        return self._agent_emitter is not None

    def event(
        self,
        event_type: str,
        payload: dict[str, Any],
        *,
        caused_by: tuple[str, ...] = (),
        structural: dict[str, Any] | None = None,
        actor: str | None = None,
    ) -> str | None:
        emitter = (
            self._environment_emitter
            if actor == "environment"
            else self._agent_emitter
        )
        if emitter is None:
            return None
        return emitter.event(
            event_type,
            payload,
            caused_by=caused_by,
            structural=structural,
        )

    def close(self) -> None:
        for emitter in (self._agent_emitter, self._environment_emitter):
            if emitter is not None:
                emitter.flush()
                emitter.close()


def _required_context_value(context: Mapping[str, Any], name: str) -> str:
    value = context.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"trace_context must contain non-empty string {name}")
    return value


__all__ = ["CraftaxTrace"]
