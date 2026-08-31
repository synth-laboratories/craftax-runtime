"""Craftax action parsing, with nothing behind it.

Deliberately free of the trace emitter and the HTTP policy: what a model's
action list means is decidable from the text and the environment's action set
alone, and a test of that should not need a pinned build-context package in
order to import.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


try:
    from parity import CRAFTAX_ACTIONS
except ImportError:
    CRAFTAX_ACTIONS = [
        "noop",
        "left",
        "right",
        "up",
        "down",
        "do",
        "sleep",
        "place_stone",
        "place_table",
        "place_furnace",
        "place_plant",
        "make_wood_pickaxe",
        "make_stone_pickaxe",
        "make_iron_pickaxe",
        "make_wood_sword",
        "make_stone_sword",
        "make_iron_sword",
        "rest",
        "descend",
        "ascend",
    ]



ACTION_WORDS = tuple(str(action) for action in CRAFTAX_ACTIONS)
# Batch bounds live with the parser that enforces them.
DEFAULT_MIN_ACTIONS_PER_CALL = 5
DEFAULT_MAX_ACTIONS_PER_CALL = 15

ACTION_RE = re.compile(
    r"\b(?:action|move|command)\s*[:=]\s*([A-Za-z0-9_\\-]+)", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedActions:
    """What the model asked for, and what will actually run.

    Seed 202's call 2 declared eleven actions including ``make_torch`` and ten
    entered execution. The eleventh was dropped by the batch cap with no record
    anywhere, so the raw assistant text and the executed trace disagreed and
    nothing said why. Declared, accepted, and dropped are now all reported.
    """

    actions: list[str]
    invalid_parse: bool
    repaired: bool
    error: str | None = None
    declared: tuple[str, ...] = ()
    dropped: tuple[str, ...] = ()
    truncation_reason: str | None = None
    #: Items the model listed that no action resolves to, with why. This is the
    #: drop that actually hit seed 202: `make_torch` is not in this env's action
    #: set, so it disappeared during resolution, before any cap applied.
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def declared_count(self) -> int:
        return len(self.declared)

    @property
    def accepted_count(self) -> int:
        return len(self.actions)


def _clean_action_token(raw: Any) -> str:
    return str(raw or "").strip().strip("\"'`.,;")


def _resolve_action(candidate: str, valid: list[str]) -> tuple[str, bool, bool]:
    cleaned = _clean_action_token(candidate)
    if cleaned in valid:
        return cleaned, False, False
    if cleaned in ACTION_WORDS:
        return valid[0], True, True
    return "", False, False


def parse_actions_text(
    raw_text: Any,
    valid_actions: list[str] | None = None,
    *,
    min_actions: int = DEFAULT_MIN_ACTIONS_PER_CALL,
    max_actions: int = DEFAULT_MAX_ACTIONS_PER_CALL,
    steps_remaining: int = DEFAULT_MAX_ACTIONS_PER_CALL,
) -> ParsedActions:
    """Parse a batched action plan from LLM JSON (code-policy shape: {"actions": [...]})."""
    valid = [
        action
        for action in ACTION_WORDS
        if not valid_actions or action in valid_actions
    ]
    if not valid:
        return ParsedActions([], True, False, "no_valid_actions")
    batch_cap = max(1, min(max_actions, steps_remaining))
    batch_floor = max(1, min(min_actions, batch_cap))
    text = str(raw_text or "").strip()
    parsed_actions: list[str] = []
    # Every item the model listed, and every one that resolved to nothing.
    # An action this environment does not have used to vanish here in silence.
    declared_items: list[str] = []
    rejected: list[tuple[str, str]] = []
    invalid_parse = False
    repaired = False
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            raw_list = parsed.get("actions")
            if isinstance(raw_list, list):
                for item in raw_list:
                    declared_items.append(_clean_action_token(item))
                    action, bad, fixed = _resolve_action(item, valid)
                    if not action:
                        rejected.append((_clean_action_token(item), "unknown_action"))
                        continue
                    invalid_parse = invalid_parse or bad
                    repaired = repaired or fixed
                    parsed_actions.append(action)
            if not parsed_actions:
                single = parsed.get("action", parsed.get("move", parsed.get("command")))
                if single is not None:
                    action, bad, fixed = _resolve_action(str(single), valid)
                    if action:
                        parsed_actions.append(action)
                        invalid_parse = bad
                        repaired = fixed
        elif isinstance(parsed, list):
            for item in parsed:
                declared_items.append(_clean_action_token(item))
                action, bad, fixed = _resolve_action(item, valid)
                if action:
                    parsed_actions.append(action)
                    invalid_parse = invalid_parse or bad
                    repaired = repaired or fixed
                else:
                    rejected.append((_clean_action_token(item), "unknown_action"))
    except json.JSONDecodeError:
        pass
    if not parsed_actions:
        single = parse_action_text(raw_text, valid_actions)
        parsed_actions = [single.action] if single.action else [valid[0]]
        invalid_parse = single.invalid_parse
        repaired = single.repaired
    declared = tuple(declared_items) if declared_items else tuple(parsed_actions)
    dropped: tuple[str, ...] = ()
    truncation_reason: str | None = None
    if len(parsed_actions) > batch_cap:
        # Truncation is a decision about the model's plan, so it is recorded as
        # one. Silently slicing the list left the raw response and the executed
        # trace disagreeing with nothing to explain the gap.
        dropped = tuple(parsed_actions[batch_cap:])
        parsed_actions = parsed_actions[:batch_cap]
        truncation_reason = (
            "batch_cap"
            if batch_cap == max_actions
            else "steps_remaining"
        )
    if len(parsed_actions) < batch_floor and parsed_actions:
        # Keep the plan the model gave; do not pad with synthetic repeats.
        pass
    if not parsed_actions:
        return ParsedActions(
            [valid[0]],
            True,
            True,
            "no_action_found",
            declared,
            dropped,
            truncation_reason,
            tuple(rejected),
        )
    return ParsedActions(
        parsed_actions,
        invalid_parse,
        repaired,
        None,
        declared,
        dropped,
        truncation_reason,
        tuple(rejected),
    )


@dataclass(frozen=True)
class ParsedAction:
    action: str
    invalid_parse: bool
    repaired: bool
    error: str | None = None


def parse_action_text(
    raw_text: Any, valid_actions: list[str] | None = None
) -> ParsedAction:
    valid = [
        action
        for action in ACTION_WORDS
        if not valid_actions or action in valid_actions
    ]
    if not valid:
        return ParsedAction("", True, False, "no_valid_actions")
    text = str(raw_text or "").strip()
    candidates: list[str] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            value = parsed.get("action", parsed.get("move", parsed.get("command")))
            if value is not None:
                candidates.append(str(value).strip())
        elif isinstance(parsed, str):
            candidates.append(parsed.strip())
    except json.JSONDecodeError:
        pass
    match = ACTION_RE.search(text)
    if match:
        candidates.append(match.group(1).strip())
    if text in ACTION_WORDS:
        candidates.append(text)
    for candidate in candidates:
        cleaned = candidate.strip().strip("\"'`.,;")
        if cleaned in valid:
            return ParsedAction(cleaned, False, False)
        if cleaned in ACTION_WORDS:
            return ParsedAction(valid[0], True, True, f"invalid_action:{cleaned}")
    return ParsedAction(valid[0], True, True, "no_action_found")
