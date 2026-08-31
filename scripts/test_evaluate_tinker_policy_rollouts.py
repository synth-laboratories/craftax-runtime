from __future__ import annotations

import asyncio
import random
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if "httpx" not in sys.modules:
    httpx = types.ModuleType("httpx")
    httpx.AsyncClient = object
    httpx.Client = object
    httpx.HTTPError = Exception
    httpx.HTTPStatusError = Exception
    httpx.Limits = object
    sys.modules["httpx"] = httpx

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object
    sys.modules["transformers"] = transformers

from evaluate_tinker_policy_rollouts import sample_policy_text, summarize_model
import evaluate_tinker_action_puzzles as action_puzzles
from evaluate_tinker_state_puzzles import (
    sample_step_count,
    score_answer,
    validate_view_sizes,
)


class _Tokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        del messages, tokenize, add_generation_prompt
        return "rendered:"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        del text, add_special_tokens
        return {"input_ids": [10, 20]}

    def decode(self, tokens: list[int]) -> str:
        del tokens
        return 'right","right","right","right","right"]}'


class _ModelInput:
    @classmethod
    def from_ints(cls, tokens: list[int]) -> list[int]:
        return tokens


class _SamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class _SampleClient:
    sampling_params: _SamplingParams | None = None

    async def sample_async(
        self,
        model_input: Any,
        *,
        num_samples: int,
        sampling_params: _SamplingParams,
    ) -> Any:
        del model_input, num_samples
        self.sampling_params = sampling_params
        sequence = types.SimpleNamespace(tokens=[30, 40], stop_reason="length")
        return types.SimpleNamespace(sequences=[sequence])


class PolicyRolloutMetricsTest(unittest.TestCase):
    def test_sampler_uses_the_requested_token_budget(self) -> None:
        client = _SampleClient()
        tinker = types.SimpleNamespace(
            ModelInput=_ModelInput,
            SamplingParams=_SamplingParams,
        )

        text, stop_reason = asyncio.run(
            sample_policy_text(
                client,
                _Tokenizer(),
                tinker,
                "prompt",
                model="openai/gpt-oss-20b",
                max_tokens=512,
            )
        )

        self.assertEqual(client.sampling_params.max_tokens, 512)
        self.assertEqual(stop_reason, "length")
        self.assertEqual(text, '{"actions":["right","right","right","right","right"]}')

    def test_summary_reports_mean_truncation_and_raw_stop_reasons(self) -> None:
        rows = [
            {
                "model": "test-model",
                "reward": 1.0,
                "achievement_count": 2,
                "decision_calls": 2,
                "parse_rate": 0.5,
                "call_truncation_rate": 0.5,
                "invalid_action_count": 0,
                "achievement_delta": {"collect_wood": 1},
                "turns": [
                    {"stop_reason": "length"},
                    {"stop_reason": "stop"},
                ],
            },
            {
                "model": "test-model",
                "reward": 3.0,
                "achievement_count": 4,
                "decision_calls": 1,
                "parse_rate": 1.0,
                "call_truncation_rate": 0.0,
                "invalid_action_count": 1,
                "achievement_delta": {"collect_stone": 1},
                "turns": [{"stop_reason": None}],
            },
        ]

        summary = summarize_model("test-model", rows)

        self.assertEqual(summary["reward_mean"], 2.0)
        self.assertEqual(summary["call_truncation_rate_mean"], 0.25)
        self.assertEqual(
            summary["stop_reason_counts"],
            {"None": 1, "length": 1, "stop": 1},
        )

    def test_observation_counts_are_scored_from_strict_text(self) -> None:
        scores = score_answer(
            {
                "observation": (
                    "<observation>\n"
                    "visible: tree=5, stone=2\n"
                    "opportunities: collect_wood=5, collect_stone=2\n"
                    "</observation>"
                )
            },
            {
                "important_terms": ["tree", "stone"],
                "important_counts": {"tree": 5, "stone": 2},
                "achievement_hints": ["collect_wood", "collect_stone"],
                "achievement_hint_counts": {
                    "collect_wood": 5,
                    "collect_stone": 2,
                },
                "immediate_achievements": [],
                "immediate_actions": [],
            },
        )

        self.assertEqual(scores["important_count_exact_recall"], 1.0)
        self.assertEqual(scores["achievement_count_exact_recall"], 1.0)

    def test_action_sampler_checkpoints_the_randomized_state(self) -> None:
        calls: list[tuple[str, str]] = []
        initial = {
            "valid_actions": ["right"],
            "observation": {"local_map": ["initial"], "achievements": []},
        }
        moved = {
            "valid_actions": ["right"],
            "observation": {"local_map": ["moved"], "achievements": ["collect_wood"]},
        }

        async def fake_http_json(_client, _base_url, method, path, payload=None, **_kwargs):
            del payload
            calls.append((method, path))
            if path == "/rollouts":
                return {"rollout_id": "rollout-1"}
            if path.endswith("/readout"):
                return moved if any(call[1].endswith("/step") for call in calls) else initial
            if path.endswith("/step"):
                return {"readout": moved, "terminated": False, "truncated": False}
            if path.endswith("/checkpoint"):
                return {"blob": "moved-checkpoint"}
            raise AssertionError((method, path))

        with patch.object(action_puzzles, "http_json", fake_http_json):
            sample = asyncio.run(
                action_puzzles.sample_checkpoint(
                    object(),
                    "http://game",
                    {"task_template": "tasks/policy_dev_template.json"},
                    seed=0,
                    max_steps_between=1,
                    view_radius=2,
                )
            )

        self.assertEqual(sample["local_map_full"], ["moved"])
        self.assertEqual(sample["checkpoint_blob"], "moved-checkpoint")
        self.assertLess(
            next(index for index, call in enumerate(calls) if call[1].endswith("/step")),
            next(
                index
                for index, call in enumerate(calls)
                if call[1].endswith("/checkpoint")
            ),
        )

    def test_state_sampler_allows_zero_steps_between_samples(self) -> None:
        self.assertEqual(sample_step_count(random.Random(7), 0), 0)

    def test_view_sizes_must_all_be_positive_and_odd(self) -> None:
        self.assertEqual(validate_view_sizes([5, 9, 13]), [5, 9, 13])
        for invalid in ([], [5, 8, 13], [0], [-3]):
            with self.subTest(invalid=invalid):
                with self.assertRaises(SystemExit):
                    validate_view_sizes(invalid)


if __name__ == "__main__":
    unittest.main()
