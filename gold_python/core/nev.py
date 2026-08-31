"""Structured NEV event log for GameBench Craftax."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EventRecord:
    step_index: int
    tick: int
    episode_id: str
    kind: str
    action: str | dict[str, Any] | None
    transition: str | None
    severity: str
    message: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "tick": self.tick,
            "episode_id": self.episode_id,
            "kind": self.kind,
            "action": self.action,
            "transition": self.transition,
            "severity": self.severity,
            "message": self.message,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EventRecord":
        return cls(
            step_index=int(data["step_index"]),
            tick=int(data.get("tick", data["step_index"])),
            episode_id=str(data["episode_id"]),
            kind=str(data["kind"]),
            action=data.get("action"),
            transition=data.get("transition"),
            severity=str(data.get("severity", "info")),
            message=str(data["message"]),
            payload=dict(data.get("payload") or {}),
        )


@dataclass
class NevLog:
    events: list[EventRecord] = field(default_factory=list)

    def append(
        self,
        *,
        step_index: int,
        episode_id: str,
        kind: str,
        message: str,
        action: str | dict[str, Any] | None = None,
        transition: str | None = None,
        severity: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> EventRecord:
        event = EventRecord(
            step_index=step_index,
            tick=step_index,
            episode_id=episode_id,
            kind=kind,
            action=action,
            transition=transition,
            severity=severity,
            message=message,
            payload=payload or {},
        )
        self.events.append(event)
        return event

    def cursor(self) -> int:
        return len(self.events)

    def export(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]

    def legacy_strings(self) -> list[str]:
        return [event.message for event in self.events]

    def export_tail(self, limit: int) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events[-max(0, int(limit)) :]]

    @classmethod
    def from_export(cls, data: list[dict[str, Any]]) -> "NevLog":
        return cls(events=[EventRecord.from_dict(item) for item in data])
