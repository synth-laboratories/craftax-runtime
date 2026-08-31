#!/usr/bin/env python3
"""Evaluate Tinker GPT-OSS models on Craftax state-perception puzzles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx


TASK_DIR = Path(__file__).resolve().parents[1]
for path in (TASK_DIR, TASK_DIR / "shared"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
DEFAULT_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1/chat/completions"
ACTION_NAMES = [
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
    "make_diamond_pickaxe",
    "make_diamond_sword",
    "make_iron_armour",
    "make_diamond_armour",
    "shoot_arrow",
    "make_arrow",
    "cast_spell",
    "place_torch",
    "read_book",
    "enchant_sword",
    "enchant_armour",
    "make_torch",
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    "enchant_bow",
]


def validate_view_sizes(view_sizes: list[int]) -> list[int]:
    if not view_sizes:
        raise SystemExit("view sizes must include at least one positive odd integer")
    invalid = [
        view_size
        for view_size in view_sizes
        if view_size <= 0 or view_size % 2 != 1
    ]
    if invalid:
        rendered = ", ".join(str(view_size) for view_size in invalid)
        raise SystemExit(f"view sizes must be positive odd integers; invalid: {rendered}")
    return view_sizes


CHAR_LABELS = {
    "P": "player",
    ".": "grass",
    ",": "path",
    "~": "water",
    "o": "stone",
    "T": "tree",
    "w": "wood",
    "S": "skeleton",
    "C": "cow",
    "c": "coal",
    "i": "iron",
    "d": "diamond",
    "s": "sapphire",
    "r": "ruby",
    "Z": "zombie",
    "a": "crafting_table",
    "F": "furnace",
    "L": "lava",
    "p": "plant",
    "R": "ripe_plant",
    ">": "down_ladder",
    "<": "up_ladder",
    "b": "book",
    "h": "chest",
    "#": "wall",
    "%": "moss_wall",
    "t": "torch",
    "u": "fountain",
    "+": "grave",
    "^": "stalagmite",
    "E": "enchantment_table",
    "?": "unknown",
}
IMPORTANT_CHARS = set("~oTwSCcidsrZaFLpR><abh#%tu+^E".replace(" ", ""))
ACHIEVEMENT_HINTS = {
    "T": "collect_wood",
    "~": "collect_drink",
    "o": "collect_stone",
    "c": "collect_coal",
    "i": "collect_iron",
    "d": "collect_diamond",
    "C": "eat_cow",
    "S": "defeat_skeleton",
    "Z": "defeat_zombie",
    "R": "eat_plant",
    "a": "craft_at_table",
    "F": "craft_at_furnace",
    ">": "descend",
    "<": "ascend",
    "b": "read_book",
    "h": "open_chest",
}
ACTION_ALIASES = {
    "do": {
        "attack",
        "chop",
        "collect",
        "cut",
        "drink",
        "eat",
        "gather",
        "hit",
        "interact",
        "kill",
        "mine",
        "use",
    },
    "descend": {"down ladder", "go down", "ladder down"},
    "ascend": {"go up", "ladder up", "up ladder"},
}
SYSTEM_PROMPT = (
    "You are solving a Craftax state-counting puzzle. Count only glyphs shown in "
    "local_map, then explain visible achievement opportunities. Reply ONLY with "
    "one JSON object with keys important_nearby, visible_counts, achievement_routes, "
    "achievement_counts, immediate_actions. Use exact ids from the prompt. Do not "
    "use aliases. Do not include prose outside JSON."
)
OBSERVATION_SYSTEM_PROMPT = (
    "You are rehearsing perception for Craftax. Describe the scenario in a natural "
    "<observation> block. Focus on what is visible, important, threatening, or "
    "achievement-relevant. Do not choose actions or plan an action sequence."
)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def client_limits() -> httpx.Limits:
    return httpx.Limits(max_connections=32, max_keepalive_connections=32)


async def http_json(
    client: httpx.AsyncClient,
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout_s: float = 60.0,
) -> dict[str, Any]:
    response = await client.request(
        method,
        urljoin(base_url.rstrip("/") + "/", path.lstrip("/")),
        json=payload,
        timeout=timeout_s,
    )
    response.raise_for_status()
    return response.json()


def wait_for_health(base_url: str, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(urljoin(base_url.rstrip("/") + "/", "health"))
                response.raise_for_status()
                payload = response.json()
            if payload.get("ok") or payload.get("status") == "ok":
                return True
        except (httpx.HTTPError, json.JSONDecodeError, TimeoutError):
            time.sleep(0.25)
    return False


def spawn_rust_service(port: int) -> subprocess.Popen[Any]:
    command = [
        sys.executable,
        str(TASK_DIR / "scripts" / "run_service.py"),
        "--lane",
        "rust",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    if not wait_for_health(f"http://127.0.0.1:{port}"):
        proc.terminate()
        raise RuntimeError(f"Craftax rust service did not become healthy on port {port}")
    return proc


def crop_map(rows: list[str], view_size: int) -> list[str]:
    if view_size > len(rows):
        raise ValueError(f"view_size={view_size} exceeds local_map rows={len(rows)}")
    center = len(rows) // 2
    radius = view_size // 2
    return [row[center - radius : center + radius + 1] for row in rows[center - radius : center + radius + 1]]


def visible_terms(rows: list[str]) -> set[str]:
    terms = set()
    for row in rows:
        for char in row:
            label = CHAR_LABELS.get(char)
            if label and char in IMPORTANT_CHARS:
                if isinstance(label, list):
                    terms.update(label)
                else:
                    terms.add(label)
    return terms


def visible_term_counts(rows: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for char in row:
            label = CHAR_LABELS.get(char)
            if label and char in IMPORTANT_CHARS:
                if isinstance(label, list):
                    counts.update(label)
                else:
                    counts[label] += 1
    return dict(sorted(counts.items()))


def visible_achievement_hints(rows: list[str]) -> set[str]:
    hints = set()
    for row in rows:
        for char in row:
            hint = ACHIEVEMENT_HINTS.get(char)
            if isinstance(hint, list):
                hints.update(hint)
            elif hint:
                hints.add(hint)
    return hints


def visible_achievement_hint_counts(rows: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        for char in row:
            hint = ACHIEVEMENT_HINTS.get(char)
            if isinstance(hint, list):
                counts.update(hint)
            elif hint:
                counts[hint] += 1
    return dict(sorted(counts.items()))


def current_achievement_counts(readout: dict[str, Any]) -> dict[str, int]:
    observation = readout.get("observation") or {}
    achievements = observation.get("achievements") or []
    if isinstance(achievements, dict):
        return {
            str(key): int(value)
            for key, value in achievements.items()
            if isinstance(value, (int, float)) and int(value) > 0
        }
    return {str(name): 1 for name in achievements}


def compact_count_text(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {flatten_text(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(flatten_text(item) for item in value)
    return str(value)


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", " ", text.lower())


def normalized_phrase_in_text(phrase: str, text: str) -> bool:
    normalized = normalize_text(phrase).strip()
    if not normalized:
        return False
    return f" {normalized} " in f" {text.strip()} "


def find_reported_count(value: Any, target: str) -> int | None:
    target_norm = normalize_text(target).strip()
    if not target_norm:
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            key_norm = normalize_text(str(key)).strip()
            if key_norm == target_norm or normalized_phrase_in_text(target_norm, key_norm):
                if isinstance(item, (int, float)):
                    return int(item)
                if isinstance(item, str):
                    match = re.search(r"-?\d+", item)
                    if match:
                        return int(match.group(0))
        for item in value.values():
            found = find_reported_count(item, target)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_reported_count(item, target)
            if found is not None:
                return found
    elif isinstance(value, str):
        target_pattern = r"[\s_]+".join(
            re.escape(part)
            for part in re.split(r"[\s_]+", target_norm)
            if part
        )
        match = re.search(
            rf"\b{target_pattern}\b\s*(?:=|:)\s*(-?\d+)\b",
            value.lower(),
        )
        if match:
            return int(match.group(1))
    return None


def parse_json_answer(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    try:
        value = json.loads(raw)
        return (value if isinstance(value, dict) else {"value": value}), True
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}, False
    try:
        value = json.loads(match.group(0))
        return (value if isinstance(value, dict) else {"value": value}), True
    except json.JSONDecodeError:
        return {}, False


def score_answer(answer: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(flatten_text(answer))
    important = set(labels["important_terms"])
    hints = set(labels["achievement_hints"])
    important_counts = dict(labels.get("important_counts") or {})
    achievement_hint_counts = dict(labels.get("achievement_hint_counts") or {})
    immediate_achievements = set(labels["immediate_achievements"])
    immediate_actions = set(labels["immediate_actions"])

    important_hit = {term for term in important if normalized_phrase_in_text(term, text)}
    hint_hit = {
        hint
        for hint in hints
        if normalized_phrase_in_text(hint, text) or normalized_phrase_in_text(hint.replace("_", " "), text)
    }
    immediate_achievement_hit = {
        item
        for item in immediate_achievements
        if normalized_phrase_in_text(item, text) or normalized_phrase_in_text(item.replace("_", " "), text)
    }
    immediate_action_hit = {item for item in immediate_actions if normalized_phrase_in_text(item, text)}
    immediate_action_semantic_hit = set(immediate_action_hit)
    for action in immediate_actions - immediate_action_hit:
        aliases = ACTION_ALIASES.get(action, set())
        if any(normalized_phrase_in_text(alias, text) for alias in aliases):
            immediate_action_semantic_hit.add(action)
    reported_counts = {
        "visible_counts": answer.get(
            "visible_counts",
            answer.get("observation", {}),
        ),
        "achievement_counts": answer.get(
            "achievement_counts",
            answer.get("observation", {}),
        ),
    }
    important_count_hit = {
        name
        for name, count in important_counts.items()
        if find_reported_count(reported_counts["visible_counts"], name) == int(count)
    }
    achievement_count_hit = {
        name
        for name, count in achievement_hint_counts.items()
        if find_reported_count(reported_counts["achievement_counts"], name) == int(count)
    }
    return {
        "important_recall": round(len(important_hit) / len(important), 4) if important else None,
        "important_count_exact_recall": round(len(important_count_hit) / len(important_counts), 4)
        if important_counts
        else None,
        "achievement_hint_recall": round(len(hint_hit) / len(hints), 4) if hints else None,
        "achievement_count_exact_recall": round(len(achievement_count_hit) / len(achievement_hint_counts), 4)
        if achievement_hint_counts
        else None,
        "immediate_achievement_recall": round(len(immediate_achievement_hit) / len(immediate_achievements), 4)
        if immediate_achievements
        else None,
        "immediate_action_recall": round(len(immediate_action_hit) / len(immediate_actions), 4)
        if immediate_actions
        else None,
        "immediate_action_semantic_recall": round(len(immediate_action_semantic_hit) / len(immediate_actions), 4)
        if immediate_actions
        else None,
        "important_hit": sorted(important_hit),
        "important_missed": sorted(important - important_hit),
        "important_count_hit": sorted(important_count_hit),
        "important_count_missed": sorted(set(important_counts) - important_count_hit),
        "achievement_hint_hit": sorted(hint_hit),
        "achievement_hint_missed": sorted(hints - hint_hit),
        "achievement_count_hit": sorted(achievement_count_hit),
        "achievement_count_missed": sorted(set(achievement_hint_counts) - achievement_count_hit),
        "immediate_achievement_hit": sorted(immediate_achievement_hit),
        "immediate_achievement_missed": sorted(immediate_achievements - immediate_achievement_hit),
        "immediate_action_hit": sorted(immediate_action_hit),
        "immediate_action_missed": sorted(immediate_actions - immediate_action_hit),
        "immediate_action_semantic_hit": sorted(immediate_action_semantic_hit),
        "immediate_action_semantic_missed": sorted(immediate_actions - immediate_action_semantic_hit),
    }


def puzzle_prompt(readout: dict[str, Any], rows: list[str], view_size: int) -> str:
    observation = dict(readout["observation"])
    inventory = observation.get("inventory") or {}
    unlocked_counts = current_achievement_counts(readout)
    inventory_text = ", ".join(
        f"{key}={value}"
        for key, value in inventory.items()
        if key not in {"potions", "learned_spells"} and value not in (0, {}, [], None)
    )
    if not inventory_text:
        inventory_text = "empty"
    nearby = observation.get("nearby_entities") or []
    nearby_text = ", ".join(
        f"{item.get('kind')}@{item.get('pos')} hp={item.get('health')}" for item in nearby
    )
    achievement_ids = sorted(set(ACHIEVEMENT_HINTS.values()))
    object_ids = sorted(
        {
            str(label)
            for char, label in CHAR_LABELS.items()
            if char in IMPORTANT_CHARS and label not in {"unknown", "grass", "path", "player"}
        }
    )
    achievement_map = ", ".join(f"{char}->{achievement}" for char, achievement in sorted(ACHIEVEMENT_HINTS.items()))
    valid_actions = [str(action) for action in readout.get("valid_actions") or ACTION_NAMES]
    return "\n".join(
        [
            f"Craftax local view ({view_size}x{view_size}).",
            f"player: pos={observation['player']['pos']} direction={observation['player']['direction']} front_tile={observation['player']['front_tile']}",
            f"inventory: {inventory_text}",
            f"unlocked_achievement_counts: {compact_count_text(unlocked_counts)}",
            "legend: P player, . grass/path, , sand/gravel, ~ water, o stone, T tree, c coal, i iron, d diamond, C cow, S skeleton, Z zombie, a table, F furnace, L lava, p plant, R ripe plant, > down ladder, < up ladder, h chest, t torch, u fountain, # wall",
            "local_map:",
            *rows,
            f"nearby_entities: {nearby_text}",
            "allowed_visible_count_keys: " + ", ".join(object_ids),
            "allowed_visible_achievement_ids: " + ", ".join(achievement_ids),
            "glyph_to_achievement_id: " + achievement_map,
            "valid_action_tokens: " + ", ".join(valid_actions),
            "Counting rules: visible_counts counts exact glyphs in local_map only. Do not count nearby_entities separately. achievement_counts counts visible glyph opportunities using glyph_to_achievement_id, not already-unlocked achievements and not tool-feasibility.",
            "Question: What important things are around you, how many are visible, and how can you get all visible achievements in this view?",
            'Reply as JSON: {"important_nearby":["tree near player"],"visible_counts":{"tree":2},"achievement_routes":["collect_wood: use do on adjacent tree"],"achievement_counts":{"collect_wood":2},"immediate_actions":["do"]}.',
            "Use only allowed_visible_count_keys in visible_counts. Use only allowed_visible_achievement_ids in achievement_counts. Use only valid_action_tokens in immediate_actions. Include only nonzero counts.",
        ]
    )


def observation_prompt(readout: dict[str, Any], rows: list[str], view_size: int) -> str:
    observation = dict(readout["observation"])
    inventory = observation.get("inventory") or {}
    unlocked_counts = current_achievement_counts(readout)
    inventory_text = ", ".join(
        f"{key}={value}"
        for key, value in inventory.items()
        if key not in {"potions", "learned_spells"} and value not in (0, {}, [], None)
    )
    if not inventory_text:
        inventory_text = "empty"
    nearby = observation.get("nearby_entities") or []
    nearby_text = ", ".join(
        f"{item.get('kind')}@{item.get('pos')} hp={item.get('health')}" for item in nearby
    )
    if not nearby_text:
        nearby_text = "none listed"
    valid_actions = [str(action) for action in readout.get("valid_actions") or ACTION_NAMES]
    return "\n".join(
        [
            f"Craftax local view ({view_size}x{view_size}).",
            f"player: pos={observation['player']['pos']} direction={observation['player']['direction']} front_tile={observation['player']['front_tile']}",
            f"inventory: {inventory_text}",
            f"unlocked_achievement_counts: {compact_count_text(unlocked_counts)}",
            "legend: P player, . grass/path, , sand/gravel, ~ water, o stone, T tree, c coal, i iron, d diamond, C cow, S skeleton, Z zombie, a table, F furnace, L lava, p plant, R ripe plant, > down ladder, < up ladder, h chest, t torch, u fountain, # wall",
            "local_map:",
            *rows,
            f"nearby_entities: {nearby_text}",
            "valid_action_tokens: " + ", ".join(valid_actions),
            "Question: What scenario are you seeing? Describe the important visible content and local achievement opportunities.",
            "Reply with one natural <observation>...</observation> block only. Recognize the situation; do not decide actions.",
        ]
    )


async def call_model(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    reasoning_effort: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    started = time.perf_counter()
    response = await client.post(
        DEFAULT_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=body,
        timeout=120.0,
    )
    latency_ms = (time.perf_counter() - started) * 1000
    if response.status_code >= 400:
        body_keys = sorted(body.keys())
        raise httpx.HTTPStatusError(
            f"{response.status_code} {response.reason_phrase}: {response.text[:1000]} "
            f"(model={model}, body_keys={body_keys})",
            request=response.request,
            response=response,
        )
    payload = response.json()
    content = payload["choices"][0]["message"].get("content", "")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "assistant_text": str(content),
        "usage": payload.get("usage", {}),
        "request_id": payload.get("id"),
        "latency_ms": latency_ms,
    }


def task_with_view_radius(suite: dict[str, Any], seed: int, view_radius: int) -> dict[str, Any]:
    task = task_for_suite_seed(suite, seed)
    task = json.loads(json.dumps(task))
    world = dict(task.get("world") or {})
    world["view_radius"] = view_radius
    task["world"] = world
    return task


def task_for_suite_seed(suite: dict[str, Any], seed: int) -> dict[str, Any]:
    template_path = TASK_DIR / str(suite.get("task_template") or "tasks/policy_dev_template.json")
    task = json.loads(template_path.read_text())
    task["seed"] = seed
    task["task_id"] = f"{task.get('task_id', 'craftax_policy')}_{seed}"
    if "max_steps" in suite:
        task["max_steps"] = int(suite["max_steps"])
    return task


async def immediate_labels(
    client: httpx.AsyncClient,
    base_url: str,
    rollout_id: str,
    readout: dict[str, Any],
) -> tuple[list[str], list[str]]:
    checkpoint = await http_json(client, base_url, "POST", f"/rollouts/{rollout_id}/checkpoint", None)
    blob = str(checkpoint["blob"])
    before = set(str(item) for item in readout["observation"].get("achievements") or [])
    sequences = [[action] for action in readout.get("valid_actions") or ACTION_NAMES]
    simulated = await http_json(
        client,
        base_url,
        "POST",
        f"/rollouts/{rollout_id}/simulate",
        {"blob": blob, "sequences": sequences},
        timeout_s=120.0,
    )
    actions = []
    achievements = set()
    for result in simulated.get("results") or []:
        after = set(str(item) for item in result.get("achievements") or [])
        gained = after - before
        if gained:
            action = str((result.get("actions") or [""])[0])
            actions.append(action)
            achievements.update(gained)
    return sorted(actions), sorted(achievements)


def sample_step_count(rng: random.Random, max_steps_between: int) -> int:
    if max_steps_between < 0:
        raise ValueError("max_steps_between must be non-negative")
    return rng.randint(1, max_steps_between) if max_steps_between else 0


async def sample_states(
    client: httpx.AsyncClient,
    base_url: str,
    suite: dict[str, Any],
    *,
    seed: int,
    states_per_seed: int,
    max_steps_between: int,
    view_radius: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    task = task_with_view_radius(suite, seed, view_radius)
    payload = await http_json(client, base_url, "POST", "/rollouts", {"task": task, "seed": seed})
    rollout_id = str(payload["rollout_id"])
    samples = []
    try:
        for sample_index in range(states_per_seed):
            readout = await http_json(client, base_url, "GET", f"/rollouts/{rollout_id}/readout")
            immediate_actions, immediate_achievements = await immediate_labels(client, base_url, rollout_id, readout)
            local_map = [str(row) for row in readout["observation"].get("local_map") or []]
            samples.append(
                {
                    "seed": seed,
                    "sample_index": sample_index,
                    "step_index": readout.get("private", {}).get("step_index"),
                    "rollout_id": rollout_id,
                    "readout": readout,
                    "local_map_full": local_map,
                    "immediate_actions": immediate_actions,
                    "immediate_achievements": immediate_achievements,
                }
            )
            valid = [str(item) for item in readout.get("valid_actions") or ["noop"]]
            step_count = sample_step_count(rng, max_steps_between)
            for _ in range(step_count):
                action = rng.choice(valid)
                payload = await http_json(
                    client,
                    base_url,
                    "POST",
                    f"/rollouts/{rollout_id}/step",
                    {"action": action},
                )
                if payload.get("terminated") or payload.get("truncated"):
                    break
                next_readout = payload.get("readout") or {}
                valid = [str(item) for item in next_readout.get("valid_actions") or valid]
            if payload.get("terminated") or payload.get("truncated"):
                break
    finally:
        try:
            await http_json(client, base_url, "DELETE", f"/rollouts/{rollout_id}", None)
        except httpx.HTTPError:
            pass
    return samples


async def evaluate_sample(
    inference_client: httpx.AsyncClient,
    api_key: str,
    model: str,
    sample: dict[str, Any],
    view_size: int,
    *,
    max_tokens: int,
    reasoning_effort: str,
    prompt_style: str,
) -> dict[str, Any]:
    rows = crop_map(sample.get("local_map_full") or sample["local_map_13"], view_size)
    labels = {
        "important_terms": sorted(visible_terms(rows)),
        "important_counts": visible_term_counts(rows),
        "achievement_hints": sorted(visible_achievement_hints(rows)),
        "achievement_hint_counts": visible_achievement_hint_counts(rows),
        "current_achievement_counts": current_achievement_counts(sample["readout"]),
        "immediate_actions": sample["immediate_actions"],
        "immediate_achievements": sample["immediate_achievements"],
    }
    prompt = (
        observation_prompt(sample["readout"], rows, view_size)
        if prompt_style == "observation"
        else puzzle_prompt(sample["readout"], rows, view_size)
    )
    assistant_text = ""
    error = None
    usage: dict[str, Any] = {}
    latency_ms = None
    parsed: dict[str, Any] = {}
    parse_ok = False
    try:
        inference = await call_model(
            inference_client,
            api_key,
            model,
            prompt,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            system_prompt=OBSERVATION_SYSTEM_PROMPT if prompt_style == "observation" else SYSTEM_PROMPT,
        )
        assistant_text = str(inference["assistant_text"])
        usage = dict(inference.get("usage") or {})
        latency_ms = float(inference["latency_ms"])
        if prompt_style == "observation":
            parsed = {"observation": assistant_text}
            parse_ok = bool(assistant_text.strip())
        else:
            parsed, parse_ok = parse_json_answer(assistant_text)
    except Exception as exc:
        error = str(exc)
    scores = score_answer(parsed, labels) if parsed else score_answer({}, labels)
    return {
        "model": model,
        "seed": sample["seed"],
        "sample_index": sample["sample_index"],
        "step_index": sample["step_index"],
        "view_size": view_size,
        "grid": rows,
        "prompt": prompt,
        "assistant_text": assistant_text,
        "parsed": parsed,
        "parse_ok": parse_ok,
        "error": error,
        "usage": usage,
        "latency_ms": latency_ms,
        "labels": labels,
        "scores": scores,
    }


def mean_present(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row["scores"][key] for row in rows if row["scores"].get(key) is not None]
    return round(statistics.mean(values), 4) if values else None


def summarize_model(model: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_view = {}
    for view_size in sorted({int(row["view_size"]) for row in rows}):
        subset = [row for row in rows if int(row["view_size"]) == view_size]
        by_view[str(view_size)] = {
            "n": len(subset),
            "parse_rate": round(sum(1 for row in subset if row["parse_ok"]) / len(subset), 4) if subset else 0.0,
            "important_recall": mean_present(subset, "important_recall"),
            "important_count_exact_recall": mean_present(subset, "important_count_exact_recall"),
            "achievement_hint_recall": mean_present(subset, "achievement_hint_recall"),
            "achievement_count_exact_recall": mean_present(subset, "achievement_count_exact_recall"),
            "immediate_achievement_recall": mean_present(subset, "immediate_achievement_recall"),
            "immediate_action_recall": mean_present(subset, "immediate_action_recall"),
            "immediate_action_semantic_recall": mean_present(subset, "immediate_action_semantic_recall"),
            "latency_ms_mean": round(statistics.mean([row["latency_ms"] for row in subset if row["latency_ms"] is not None]), 2)
            if any(row["latency_ms"] is not None for row in subset)
            else None,
        }
    return {
        "model": model,
        "n": len(rows),
        "parse_rate": round(sum(1 for row in rows if row["parse_ok"]) / len(rows), 4) if rows else 0.0,
        "prompt_tokens": sum(int(row.get("usage", {}).get("prompt_tokens", 0) or 0) for row in rows),
        "completion_tokens": sum(int(row.get("usage", {}).get("completion_tokens", 0) or 0) for row in rows),
        "error_count": sum(1 for row in rows if row.get("error")),
        "by_view_size": by_view,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key and not args.sample_only:
        raise SystemExit("TINKER_API_KEY is required")
    models = [part.strip() for part in args.models.split(",") if part.strip()] or DEFAULT_MODELS
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    view_sizes = validate_view_sizes(
        [int(part.strip()) for part in args.view_sizes.split(",") if part.strip()]
    )
    max_view = max(view_sizes)
    suite = json.loads(Path(args.suite).read_text())
    proc: subprocess.Popen[Any] | None = None
    base_url = args.base_url
    started = time.perf_counter()
    try:
        if not base_url:
            proc = spawn_rust_service(args.port)
            base_url = f"http://127.0.0.1:{args.port}"
        samples = []
        async with httpx.AsyncClient(limits=client_limits(), timeout=60.0) as game_client:
            for seed in seeds:
                log(f"sampling seed={seed}")
                samples.extend(
                    await sample_states(
                        game_client,
                        base_url,
                        suite,
                        seed=seed,
                        states_per_seed=args.states_per_seed,
                        max_steps_between=args.max_steps_between,
                        view_radius=max_view // 2,
                    )
                )
        rows = []
        if not args.sample_only:
            async with httpx.AsyncClient(limits=client_limits(), timeout=120.0) as inference_client:
                for model in models:
                    for sample in samples:
                        for view_size in view_sizes:
                            log(
                                f"eval model={model} seed={sample['seed']} sample={sample['sample_index']} view={view_size}"
                            )
                            rows.append(
                                await evaluate_sample(
                                    inference_client,
                                    api_key,
                                    model,
                                    sample,
                                    view_size,
                                    max_tokens=args.max_tokens,
                                    reasoning_effort=args.reasoning_effort,
                                    prompt_style=args.prompt_style,
                                )
                            )
        report = {
            "schema": "gamebench.craftax.tinker_state_puzzle_eval.v1",
            "engine_lane": "rust",
            "inference_provider": "tinker",
            "models": models,
            "seeds": seeds,
            "view_sizes": view_sizes,
            "states_per_seed": args.states_per_seed,
            "sample_count": len(samples),
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort or None,
            "prompt_style": args.prompt_style,
            "sample_only": bool(args.sample_only),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "summaries": [summarize_model(model, [row for row in rows if row["model"] == model]) for model in models],
            "samples": [
                {
                    "seed": sample["seed"],
                    "sample_index": sample["sample_index"],
                    "step_index": sample["step_index"],
                    "immediate_actions": sample["immediate_actions"],
                    "immediate_achievements": sample["immediate_achievements"],
                    "local_map_full": sample.get("local_map_full") or sample["local_map_13"],
                }
                for sample in samples
            ],
            "results": rows,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="101,102,103")
    parser.add_argument("--states-per-seed", type=int, default=3)
    parser.add_argument("--max-steps-between", type=int, default=12)
    parser.add_argument("--view-sizes", default="5,9,13")
    parser.add_argument("--suite", default=str(TASK_DIR / "defaults" / "policy_sweep" / "policy_dev_v1.json"))
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--reasoning-effort", default="")
    parser.add_argument("--prompt-style", choices=["json", "observation"], default="json")
    parser.add_argument("--port", type=int, default=19113)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument(
        "--output",
        default=str(TASK_DIR / "reports" / "state_puzzles" / "tinker_gpt_oss_state_puzzles.json"),
    )
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    print(
        json.dumps(
            {
                "models": report["models"],
                "sample_count": report["sample_count"],
                "view_sizes": report["view_sizes"],
                "summaries": report["summaries"],
                "elapsed_s": report["elapsed_s"],
                "output": args.output,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
