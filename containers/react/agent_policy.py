"""OpenAI-compatible Craftax ReAct policy."""

from __future__ import annotations

import json
import hashlib
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from containers.react.trace_emitter import CraftaxTrace


# Parsing lives in `action_parsing`, which depends on nothing but the action
# set. Re-exported here so existing callers keep one import.
from containers.react.action_parsing import (  # noqa: F401
    ACTION_RE,
    ACTION_WORDS,
    CRAFTAX_ACTIONS,
    DEFAULT_MAX_ACTIONS_PER_CALL,
    DEFAULT_MIN_ACTIONS_PER_CALL,
    ParsedAction,
    ParsedActions,
    _clean_action_token,
    _resolve_action,
    parse_action_text,
    parse_actions_text,
)

DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_SYSTEM_PROMPT = (
    "You are playing GameBench Craftax rust gold. Each response must plan a short sequence of "
    "valid actions to execute in order before the next observation. Prioritize survival, collect "
    "wood and stone, craft basic tools, recover coal and iron, and make durable achievement progress. "
    'Reply with JSON only, for example {"actions":["do","right","do","left","do"]}.'
)


@dataclass
class AgentPolicyConfig:
    policy_id: str = "craftax_react_groq_v1"
    inference_url: str = DEFAULT_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    # 512 is measured adequate for the providers this policy defaults to: Groq
    # gpt-oss-120b answers in ~180-230 tokens and Gemini flash-lite in ~11. Qwen3.5
    # instead reasons in prose on this plain-JSON prompt and truncates at 512 and at
    # 2048 alike, so it needs the tool-calling policy variant, not a larger budget.
    max_tokens: int = 512
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    use_lm: bool = True
    provider: str = "groq"
    min_actions_per_call: int = DEFAULT_MIN_ACTIONS_PER_CALL
    max_actions_per_call: int = DEFAULT_MAX_ACTIONS_PER_CALL

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AgentPolicyConfig":
        mode = str(raw.get("credential_mode", "")).strip().lower()
        proxied = mode in {"workshop_proxy", "proxy"}
        if proxied:
            inference_url = str(raw.get("inference_url") or "").strip()
            if not inference_url:
                raise ValueError("inference_url is required when credential_mode=workshop_proxy")
            lowered = inference_url.lower()
            if "api.openai.com" in lowered or "127.0.0.1" in lowered or "localhost" in lowered:
                raise ValueError("workshop_proxy inference_url must be the Workshop container proxy")
            if not inference_url.rstrip("/").endswith("/chat/completions"):
                inference_url = inference_url.rstrip("/") + "/chat/completions"
            api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "workshop-proxy"
            return cls(
                policy_id=str(raw.get("policy_id", "craftax_react_workshop_proxy_v1")),
                inference_url=inference_url,
                api_key=api_key,
                model=str(raw.get("model", "gpt-4.1-nano")),
                temperature=float(raw.get("temperature", 0.0)),
                max_tokens=max(int(raw.get("max_tokens", 512)), 32),
                system_prompt=str(raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
                use_lm=bool(raw.get("use_lm", True)),
                provider=str(raw.get("provider", "openai")).strip().lower(),
                min_actions_per_call=max(
                    int(raw.get("min_actions_per_call", DEFAULT_MIN_ACTIONS_PER_CALL)), 1
                ),
                max_actions_per_call=max(
                    int(raw.get("max_actions_per_call", DEFAULT_MAX_ACTIONS_PER_CALL)), 1
                ),
            )
        provider = str(raw.get("provider", "groq")).strip().lower()
        default_url = DEFAULT_URL
        if provider == "deepseek":
            default_url = "https://api.deepseek.com/chat/completions"
        elif provider == "gemini":
            default_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif provider == "openai":
            default_url = "https://api.openai.com/v1/chat/completions"
        default_api_key_env = (
            "GROQ_API_KEY" if provider == "groq" else "DEEPSEEK_API_KEY"
        )
        if provider == "gemini":
            default_api_key_env = "GEMINI_API_KEY"
        elif provider == "openai":
            default_api_key_env = "OPENAI_API_KEY"
        api_key_env = str(raw.get("api_key_env", default_api_key_env))
        if provider == "gemini":
            api_key = (
                str(raw.get("api_key", "")).strip()
                or os.environ.get(api_key_env, "").strip()
                or os.environ.get("GEMINI_API_KEY", "").strip()
                or os.environ.get("GOOGLE_API_KEY", "").strip()
            )
        else:
            api_key = (
                str(raw.get("api_key", "")).strip()
                or os.environ.get(api_key_env, "").strip()
                or os.environ.get("GROQ_API_KEY", "").strip()
                or os.environ.get("DEEPSEEK_API_KEY", "").strip()
                or os.environ.get("OPENAI_API_KEY", "").strip()
            )
        default_model = DEFAULT_MODEL
        if provider == "deepseek":
            default_model = "deepseek-v4-flash"
        elif provider == "gemini":
            default_model = "gemini-3.1-flash-lite"
        return cls(
            policy_id=str(raw.get("policy_id", "craftax_react_groq_v1")),
            inference_url=str(
                raw.get("inference_url", raw.get("base_url", default_url))
            ),
            api_key=api_key,
            model=str(raw.get("model", default_model)),
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=max(int(raw.get("max_tokens", 512)), 32),
            system_prompt=str(raw.get("system_prompt", DEFAULT_SYSTEM_PROMPT)),
            use_lm=bool(raw.get("use_lm", True)),
            provider=provider,
            min_actions_per_call=max(
                int(raw.get("min_actions_per_call", DEFAULT_MIN_ACTIONS_PER_CALL)), 1
            ),
            max_actions_per_call=max(
                int(raw.get("max_actions_per_call", DEFAULT_MAX_ACTIONS_PER_CALL)), 1
            ),
        )


@dataclass
class AgentTurnResult:
    actions: list[str]
    assistant_text: str
    invalid_parse: bool
    repaired: bool
    # What the model asked for, beside what will run. Seed 202 declared eleven
    # actions and executed ten, and no record anywhere said so.
    declared_actions: list[str] = field(default_factory=list)
    dropped_actions: list[str] = field(default_factory=list)
    rejected_actions: list[dict[str, str]] = field(default_factory=list)
    truncation_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None
    model: str | None = None
    finish_reason: str | None = None
    request_event_id: str | None = None
    response_event_id: str | None = None
    proposal_event_id: str | None = None

    @property
    def action(self) -> str:
        return self.actions[0] if self.actions else "noop"

    @property
    def truncated(self) -> bool:
        """The provider stopped at the token budget, so no action batch was emitted."""
        return self.finish_reason == "length"



async def chat_completion(
    config: AgentPolicyConfig,
    prompt: str,
    *,
    trace: CraftaxTrace | None = None,
    llm_call: int | None = None,
) -> dict[str, Any]:
    call_correlation_id = f"craftax-call-{llm_call or 0}-{uuid.uuid4().hex}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "x-synth-call-correlation-id": call_correlation_id,
    }
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    if config.provider == "deepseek":
        body["thinking"] = {"type": "disabled"}
    request_event_id = (
        trace.event(
            "agent.model_call_intent",
            {
                "provider": config.provider,
                "model": config.model,
                "call_correlation_id": call_correlation_id,
                "prompt_digest": "sha256:"
                + hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "llm_call": llm_call,
            },
            structural={"llm_call": llm_call},
        )
        if trace is not None
        else None
    )
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(config.inference_url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
    choice = payload["choices"][0]
    content = choice["message"].get("content", "")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    response_event_id = (
        trace.event(
            "agent.model_call_observed",
            {
                "provider": config.provider,
                "model": config.model,
                "call_correlation_id": call_correlation_id,
                "provider_request_id": payload.get("id"),
                "finish_reason": str(choice.get("finish_reason") or "") or None,
                "usage": payload.get("usage", {}),
                "llm_call": llm_call,
            },
            caused_by=tuple(item for item in (request_event_id,) if item),
            structural={"llm_call": llm_call},
        )
        if trace is not None
        else None
    )
    return {
        "assistant_text": str(content),
        "usage": payload.get("usage", {}),
        "request_id": payload.get("id"),
        # A response cut off at max_tokens cannot carry an action batch. Without
        # this the caller sees only invalid_parse and blames the prompt.
        "finish_reason": str(choice.get("finish_reason") or "") or None,
        "request_event_id": request_event_id,
        "response_event_id": response_event_id,
    }


class AgentPolicy:
    ACTOR = "craftax_agent"

    def __init__(self, config: AgentPolicyConfig, *, trace: CraftaxTrace | None = None) -> None:
        self.config = config
        self.trace = trace

    async def choose_action(
        self,
        *,
        readout: dict[str, Any],
        objective: str,
        action_history: list[str],
        steps_remaining: int,
        llm_calls_remaining: int,
        llm_call: int | None = None,
    ) -> AgentTurnResult:
        if steps_remaining <= 0 or llm_calls_remaining <= 0:
            raise RuntimeError(
                "craftax policy called with no remaining steps or llm calls"
            )
        valid_actions = list(readout.get("valid_actions") or ACTION_WORDS)
        batch_cap = max(1, min(self.config.max_actions_per_call, steps_remaining))
        batch_floor = max(1, min(self.config.min_actions_per_call, batch_cap))
        observation_text = str(readout.get("observation_text") or "")
        prompt = "\n".join(
            [
                objective,
                "",
                observation_text,
                "",
                f"last_actions={json.dumps(action_history[-16:])}",
                f"steps_remaining={steps_remaining}",
                f"llm_calls_remaining={llm_calls_remaining}",
                f"valid_actions={', '.join(valid_actions)}",
                f"Plan {batch_floor}-{batch_cap} valid actions to execute sequentially before the next observation.",
                'Reply with JSON only, for example {"actions":["do","right","do","left","do"]}.',
            ]
        )
        if not self.config.use_lm:
            raise RuntimeError(
                "craftax react policy requires use_lm=true for GELO validation"
            )
        if not self.config.api_key:
            raise RuntimeError(
                f"missing_api_key provider={self.config.provider} policy={self.config.policy_id}"
            )
        if self.trace is not None and len(action_history) > 16:
            self.trace.event(
                "agent.context_compacted",
                {
                    "strategy": "tail_window",
                    "retained_action_count": 16,
                    "dropped_action_count": len(action_history) - 16,
                    "replacement": "last_actions",
                },
                structural={"llm_call": llm_call},
            )
        inference = await chat_completion(
            self.config,
            prompt,
            trace=self.trace,
            llm_call=llm_call,
        )
        parsed = parse_actions_text(
            inference["assistant_text"],
            valid_actions,
            min_actions=batch_floor,
            max_actions=batch_cap,
            steps_remaining=steps_remaining,
        )
        proposal_event_id = (
            self.trace.event(
                "agent.action_proposed",
                {
                    "actions": list(parsed.actions),
                    "assistant_text": inference["assistant_text"],
                    "invalid_parse": parsed.invalid_parse,
                    "repaired": parsed.repaired,
                    "parse_error": parsed.error,
                    # Declared vs accepted, always both. A reader comparing the
                    # raw response with the executed trace must not have to
                    # infer that the difference was a cap rather than a bug.
                    "declared_actions": list(parsed.declared),
                    "declared_count": parsed.declared_count,
                    "accepted_count": parsed.accepted_count,
                    "dropped_actions": list(parsed.dropped),
                    "dropped_count": len(parsed.dropped),
                    "truncation_reason": parsed.truncation_reason,
                    "rejected_actions": [
                        {"action": action, "reason": reason}
                        for action, reason in parsed.rejected
                    ],
                    "batch_cap": batch_cap,
                },
                caused_by=tuple(
                    item for item in (inference.get("response_event_id"),) if item
                ),
            )
            if self.trace is not None
            else None
        )
        return AgentTurnResult(
            actions=list(parsed.actions),
            declared_actions=list(parsed.declared),
            dropped_actions=list(parsed.dropped),
            rejected_actions=[
                {"action": action, "reason": reason} for action, reason in parsed.rejected
            ],
            truncation_reason=parsed.truncation_reason,
            assistant_text=inference["assistant_text"],
            invalid_parse=parsed.invalid_parse,
            repaired=parsed.repaired,
            usage=dict(inference.get("usage", {})),
            request_id=inference.get("request_id"),
            model=self.config.model,
            finish_reason=inference.get("finish_reason"),
            request_event_id=inference.get("request_event_id"),
            response_event_id=inference.get("response_event_id"),
            proposal_event_id=proposal_event_id,
        )
