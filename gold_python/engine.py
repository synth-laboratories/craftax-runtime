"""Independent symbolic Craftax engine for GameBench."""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any

from core.checkpoint import NEV_TAIL_EVENTS, decode_checkpoint, digest_json, encode_checkpoint
from core.nev import NevLog
from parity import (
    BLOCK_TYPES,
    CRAFTAX_ACHIEVEMENTS,
    CRAFTAX_ACTIONS,
    FLOOR_MOB_SPAWN_CHANCE,
    FLOOR_MOBS,
    ITEM_TYPES,
    LEVEL_ACHIEVEMENTS,
    MELEE_MOBS,
    MOB_CLASSES,
    MOB_ACHIEVEMENTS,
    MOB_ALIASES,
    MOB_HEALTH,
    MOB_NAMES,
    MOB_TYPE_COLLISION_MAPPING,
    MOB_TYPE_DEFENSE_MAPPING,
    PASSIVE_MOBS,
    PROJECTILE_TYPES,
    PROJECTILE_DAMAGE_TYPE_IDS,
    MOB_TYPE_IDS,
    MOB_TYPE_DAMAGE_MAPPING,
    RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_IDS,
    RANGED_MOBS,
    TILE_ALIASES,
)
from state import PrivateState, PublicState
from task_resolve import ResolvedTask, resolve_task, stable_hash
from worldgen import LEVEL_STACK, generate_world_layout


ACTION_NAMES = CRAFTAX_ACTIONS
MONSTERS_KILLED_TO_CLEAR_LEVEL = 8
BOSS_FIGHT_SPAWN_TURNS = 7
DAY_LENGTH = 300
ACTION_ALIASES = {
    "move_left": "left",
    "move_right": "right",
    "move_up": "up",
    "move_down": "down",
    "wait": "noop",
    "cast_fireball": "cast_spell",
    "cast_iceball": "cast_spell",
}
DIRS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
}
CARDINAL_DIRECTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
PASSIVE_DIRECTIONS = CARDINAL_DIRECTIONS + [(0, 0), (0, 0), (0, 0), (0, 0)]
WALKABLE = {
    "grass",
    "path",
    "sand",
    "ladder_down",
    "ladder_up",
    "torch",
    "fire_grass",
    "ice_grass",
    "gravel",
}
ITEM_OVERLAYS = set(ITEM_TYPES) - {"none"}
VALID_BLOCKS = set(BLOCK_TYPES)
VALID_ITEMS = set(ITEM_TYPES)
VALID_POSE_TILES = VALID_BLOCKS | {item for item in VALID_ITEMS if item != "none"} | {"out_of_bounds"}
VALID_PROJECTILE_OWNERS = {"player", "mob"}
VALID_DIRECTIONS = set(CARDINAL_DIRECTIONS)
PLAYER_PROJECTILE_KINDS = {"arrow2", "fireball"}
MOB_PROJECTILE_KINDS = {PROJECTILE_TYPES[index] for index in RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_IDS}
VALID_NEV_EVENT_KINDS = {
    "achievement_unlocked",
    "action_applied",
    "checkpoint_cadence",
    "combat",
    "death",
    "entity_transition",
    "episode_truncated",
    "floor_transition",
    "projectile_transition",
    "resource_delta",
    "reward_delta",
    "rule_violation",
    "state_transition",
    "task_resolved",
    "terminal",
    "terminal_success",
}
BASE_WALKABLE = WALKABLE - ITEM_OVERLAYS
CAN_PLACE_ITEM_ON = {"grass", "sand", "path", "fire_grass", "ice_grass"}
SOLID = {
    "water",
    "stone",
    "tree",
    "coal",
    "iron",
    "diamond",
    "sapphire",
    "ruby",
    "chest",
    "wall",
    "wall_moss",
    "stalagmite",
    "plant",
    "ripe_plant",
    "crafting_table",
    "furnace",
    "fountain",
    "lava",
    "darkness",
    "fire_tree",
    "ice_shrub",
    "enchantment_table_fire",
    "enchantment_table_ice",
    "necromancer",
    "necromancer_vulnerable",
    "grave",
    "grave2",
    "grave3",
}
STATIC_SOLID_BLOCKS = SOLID - {"water", "lava", "darkness", "ice_shrub", "necromancer_vulnerable"}
RESOURCE_TILE = {
    "tree": ("wood", "collect_wood", 0),
    "fire_tree": ("wood", "collect_wood", 0),
    "ice_shrub": ("wood", "collect_wood", 0),
    "stone": ("stone", "collect_stone", 1),
    "stalagmite": ("stone", "collect_stone", 1),
    "coal": ("coal", "collect_coal", 1),
    "iron": ("iron", "collect_iron", 2),
    "diamond": ("diamond", "collect_diamond", 3),
    "sapphire": ("sapphire", "collect_sapphire", 4),
    "ruby": ("ruby", "collect_ruby", 4),
}
POTION_ACTIONS = {
    "drink_potion_red": "red",
    "drink_potion_green": "green",
    "drink_potion_blue": "blue",
    "drink_potion_pink": "pink",
    "drink_potion_cyan": "cyan",
    "drink_potion_yellow": "yellow",
}
HOSTILE_MOBS = set(MOB_NAMES) - set(PASSIVE_MOBS)
TIER_NAMES = ["none", "wood", "stone", "iron", "diamond"]
POTION_COLORS = ("red", "green", "blue", "pink", "cyan", "yellow")
CHEST_ORE_LOOT = (
    ("coal", 0.3, 1, 4),
    ("iron", 0.3, 1, 3),
    ("diamond", 0.15, 1, 2),
    ("sapphire", 0.125, 1, 2),
    ("ruby", 0.125, 1, 2),
)
CHEST_TOOL_LEVELS = (1, 2, 3, 4)
CHEST_TOOL_LEVEL_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
CHEST_TOOL_LOOT = tuple(zip(CHEST_TOOL_LEVELS, CHEST_TOOL_LEVEL_WEIGHTS, strict=True))
MAX_GROWING_PLANTS = 10


class CraftaxInvariantError(RuntimeError):
    """Raised when the engine reaches an impossible Craftax runtime state."""


def default_inventory() -> dict[str, Any]:
    return {
        "wood": 0,
        "stone": 0,
        "coal": 0,
        "iron": 0,
        "diamond": 0,
        "sapling": 0,
        "ruby": 0,
        "sapphire": 0,
        "pickaxe": 0,
        "sword": 0,
        "bow": 0,
        "arrows": 0,
        "torches": 0,
        "books": 0,
        "armour": [0, 0, 0, 0],
        "health": 9,
        "food": 9,
        "drink": 9,
        "energy": 9,
        "mana": 9,
        "xp": 0,
        "dexterity": 1,
        "strength": 1,
        "intelligence": 1,
        "potions": {"red": 0, "green": 0, "blue": 0, "pink": 0, "cyan": 0, "yellow": 0},
        "learned_spells": [],
        "sword_enchantment": "none",
        "bow_enchantment": "none",
        "armour_enchantments": ["none", "none", "none", "none"],
        "boss_progress": 0,
    }


@dataclass
class Entity:
    id: str
    kind: str
    pos: tuple[int, int]
    health: float
    level: int = 0
    mob_class: str = ""
    attack_cooldown: int = 0
    mask: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "class": self.mob_class or _mob_class(self.kind),
            "level": self.level,
            "pos": [self.pos[0], self.pos[1]],
            "health": self.health,
            "attack_cooldown": self.attack_cooldown,
            "mask": self.mask,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        kind = normalize_mob(str(data["kind"]))
        return cls(
            id=str(data["id"]),
            kind=kind,
            pos=_pos(data["pos"], "entity.pos"),
            health=float(data["health"]),
            level=_strict_int(data.get("level", 0), "entity.level"),
            mob_class=str(data.get("class") or data.get("mob_class") or _mob_class(kind)),
            attack_cooldown=_strict_int(data.get("attack_cooldown", 0), "entity.attack_cooldown"),
            mask=_strict_bool(data.get("mask", True), "entity.mask"),
        )


@dataclass
class Projectile:
    id: str
    kind: str
    pos: tuple[int, int]
    direction: tuple[int, int]
    level: int = 0
    owner: str = "player"
    mask: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "owner": self.owner,
            "level": self.level,
            "pos": [self.pos[0], self.pos[1]],
            "direction": [self.direction[0], self.direction[1]],
            "mask": self.mask,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Projectile":
        return cls(
            id=str(data["id"]),
            kind=str(data["kind"]),
            owner=str(data.get("owner", "player")),
            level=_strict_int(data.get("level", 0), "projectile.level"),
            pos=_pos(data["pos"], "projectile.pos"),
            direction=_dir(data["direction"], "projectile.direction"),
            mask=_strict_bool(data.get("mask", True), "projectile.mask"),
        )


@dataclass
class CraftaxWorld:
    width: int
    height: int
    levels: int
    max_steps: int
    seed: int
    maps: list[list[list[str]]]
    item_maps: list[list[list[str]]]
    light_maps: list[list[list[float]]]
    down_ladders: list[tuple[int, int]]
    up_ladders: list[tuple[int, int]]
    chests_opened: list[bool]
    monsters_killed: list[int]
    potion_mapping: list[int]
    player_pos: tuple[int, int]
    player_direction: tuple[int, int] = (0, 1)
    player_level: int = 0
    inventory: dict[str, Any] = field(default_factory=default_inventory)
    achievements: dict[str, int] = field(default_factory=dict)
    entities: list[Entity] = field(default_factory=list)
    player_projectiles: list[Projectile] = field(default_factory=list)
    mob_projectiles: list[Projectile] = field(default_factory=list)
    growing_plants: list[dict[str, Any]] = field(default_factory=list)
    is_sleeping: bool = False
    is_resting: bool = False
    player_recover: float = 0.0
    player_hunger: float = 0.0
    player_thirst: float = 0.0
    player_fatigue: float = 0.0
    player_recover_mana: float = 0.0
    boss_timesteps_to_spawn_this_round: int = 0
    light_level: float = 1.0
    timestep: int = 0
    rng_state: object | None = None

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "levels": self.levels,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "maps": self.maps,
            "item_maps": self.item_maps,
            "light_maps": self.light_maps,
            "down_ladders": [[x, y] for x, y in self.down_ladders],
            "up_ladders": [[x, y] for x, y in self.up_ladders],
            "chests_opened": list(self.chests_opened),
            "monsters_killed": list(self.monsters_killed),
            "potion_mapping": list(self.potion_mapping),
            "player_pos": [self.player_pos[0], self.player_pos[1]],
            "player_direction": [self.player_direction[0], self.player_direction[1]],
            "player_level": self.player_level,
            "inventory": copy.deepcopy(self.inventory),
            "achievements": dict(self.achievements),
            "entities": [entity.to_dict() for entity in self.entities],
            "player_projectiles": [projectile.to_dict() for projectile in self.player_projectiles],
            "mob_projectiles": [projectile.to_dict() for projectile in self.mob_projectiles],
            "growing_plants": copy.deepcopy(self.growing_plants),
            "is_sleeping": self.is_sleeping,
            "is_resting": self.is_resting,
            "player_recover": self.player_recover,
            "player_hunger": self.player_hunger,
            "player_thirst": self.player_thirst,
            "player_fatigue": self.player_fatigue,
            "player_recover_mana": self.player_recover_mana,
            "boss_timesteps_to_spawn_this_round": self.boss_timesteps_to_spawn_this_round,
            "light_level": self.light_level,
            "timestep": self.timestep,
            "rng_state": self.rng_state,
        }

    @classmethod
    def from_checkpoint(cls, data: dict[str, Any]) -> "CraftaxWorld":
        maps = [[[str(cell) for cell in row] for row in level] for level in data["maps"]]
        item_maps = _normalize_item_maps(data.get("item_maps"), maps)
        levels = _strict_int(data["levels"], "world.levels")
        timestep = _strict_int(data["timestep"], "world.timestep")
        return cls(
            width=_strict_int(data["width"], "world.width"),
            height=_strict_int(data["height"], "world.height"),
            levels=levels,
            max_steps=_strict_int(data["max_steps"], "world.max_steps"),
            seed=_strict_int(data["seed"], "world.seed"),
            maps=maps,
            item_maps=item_maps,
            light_maps=_normalize_light_maps(data.get("light_maps"), levels, _strict_int(data["width"], "world.width"), _strict_int(data["height"], "world.height")),
            down_ladders=_normalize_positions(data.get("down_ladders"), levels, (-1, -1)),
            up_ladders=_normalize_positions(data.get("up_ladders"), levels, (-1, -1)),
            chests_opened=_normalize_bool_vector(data.get("chests_opened"), levels, "chests_opened"),
            monsters_killed=_normalize_int_vector(data.get("monsters_killed"), levels, "monsters_killed"),
            potion_mapping=_normalize_int_vector(data.get("potion_mapping", list(range(6))), 6, "potion_mapping"),
            player_pos=_pos(data["player_pos"], "player_pos"),
            player_direction=_dir(data["player_direction"], "player_direction"),
            player_level=_strict_int(data["player_level"], "player_level"),
            inventory=normalize_inventory(copy.deepcopy(data["inventory"])),
            achievements={str(key): _strict_int(value, f"achievements[{key}]") for key, value in data.get("achievements", {}).items()},
            entities=[Entity.from_dict(item) for item in data.get("entities", [])],
            player_projectiles=[Projectile.from_dict(item) for item in data.get("player_projectiles", [])],
            mob_projectiles=[Projectile.from_dict(item) for item in data.get("mob_projectiles", [])],
            growing_plants=copy.deepcopy(data.get("growing_plants", [])),
            is_sleeping=_strict_bool(data.get("is_sleeping", False), "is_sleeping"),
            is_resting=_strict_bool(data.get("is_resting", False), "is_resting"),
            player_recover=_strict_finite_number(data.get("player_recover", 0.0), "world.player_recover"),
            player_hunger=_strict_finite_number(data.get("player_hunger", 0.0), "world.player_hunger"),
            player_thirst=_strict_finite_number(data.get("player_thirst", 0.0), "world.player_thirst"),
            player_fatigue=_strict_finite_number(data.get("player_fatigue", 0.0), "world.player_fatigue"),
            player_recover_mana=_strict_finite_number(data.get("player_recover_mana", 0.0), "world.player_recover_mana"),
            boss_timesteps_to_spawn_this_round=_strict_int(data.get("boss_timesteps_to_spawn_this_round", 0), "world.boss_timesteps_to_spawn_this_round"),
            light_level=_strict_finite_number(data.get("light_level", _calculate_light_level(timestep, DAY_LENGTH)), "world.light_level"),
            timestep=timestep,
            rng_state=data.get("rng_state"),
        )


class CraftaxEngine:
    ENV_FAMILY = "craftax-singleplayer"

    def __init__(self) -> None:
        self.resolved: ResolvedTask | None = None
        self.world: CraftaxWorld | None = None
        self.public = PublicState({}, (0, 0), 0, default_inventory(), {}, False)
        self.private = PrivateState()
        self.nev = NevLog()
        self._rng = random.Random(0)

    def reset(self, resolved: ResolvedTask) -> PublicState:
        self.resolved = resolved
        self._rng = random.Random(resolved.seed)
        self.world = self._make_world(resolved)
        seeded = {name for name, count in self.world.achievements.items() if count > 0}
        self.private = PrivateState(
            episode_id=resolved.episode_id,
            task_id=resolved.task_id,
            scenario_id=resolved.scenario_id,
            seed=resolved.seed,
            config_hash=resolved.config_hash,
            achievements=seeded,
        )
        self._assert_runtime_invariants("reset")
        self.public = self._public_state(done=False)
        self.nev = NevLog()
        self.nev.append(
            step_index=0,
            episode_id=resolved.episode_id,
            kind="task_resolved",
            transition="reset",
            message=f"TaskResolved({resolved.scenario_id},{resolved.config_hash})",
            payload={"resolved": resolved.to_dict()},
        )
        return self.public

    def reset_from_task(self, task: dict[str, Any], seed_override: int | None = None) -> PublicState:
        return self.reset(resolve_task(task, seed_override=seed_override))

    def step(self, raw_action: str | dict[str, Any]) -> dict[str, Any]:
        action = normalize_action(raw_action)
        if self.world is None or self.resolved is None:
            raise RuntimeError("engine must be reset before step")
        if self.private.terminated or self.private.truncated:
            self._reject_terminal(action)
            return self.symbolic_readout()
        requested_action = action
        invalid_action_code = "unknown_action" if action not in ACTION_NAMES else None

        intrinsic_wake_action: str | None = None
        if invalid_action_code is None and bool(self.resolved.rules.get("homeostasis", False)) and (self.world.is_sleeping or self.world.is_resting):
            continuing_sleep = self.world.is_sleeping and action in {"noop", "sleep"}
            continuing_rest = self.world.is_resting and action in {"noop", "rest"}
            if continuing_sleep or continuing_rest:
                action = "noop"
            else:
                intrinsic_wake_action = requested_action
                action = "noop"

        before_inventory = copy.deepcopy(self.world.inventory)
        before_achievements = dict(self.world.achievements)
        before_health = float(self.world.inventory.get("health", 0.0))
        before_pos = self.world.player_pos
        before_level = self.world.player_level
        before_tile = self.tile_at(self.world.player_pos)
        self.world.timestep += 1
        self.private.step_index = self.world.timestep
        self.private.reward_last = 0.0
        self._apply_step_reward()

        if intrinsic_wake_action is not None:
            self._wake_from_intrinsic(intrinsic_wake_action)
        elif invalid_action_code is not None:
            self._reject(action, invalid_action_code)
        elif action in DIRS:
            self._move(action)
        elif action == "do":
            self._do()
        elif action in {"sleep", "rest"}:
            if bool(self.resolved.rules.get("homeostasis", False)):
                self._append_action(action, "intrinsic_mode_request", {"sleeping": self.world.is_sleeping, "resting": self.world.is_resting})
            else:
                self._recover(action)
        elif action == "descend":
            self._descend()
        elif action == "ascend":
            self._ascend()
        elif action.startswith("place_"):
            self._place(action)
        elif action.startswith("make_"):
            self._craft(action)
        elif action == "shoot_arrow":
            self._shoot_arrow()
        elif action == "cast_spell":
            self._cast_spell()
        elif action in POTION_ACTIONS:
            self._drink_potion(POTION_ACTIONS[action])
        elif action == "read_book":
            self._read_book()
        elif action in {"enchant_sword", "enchant_armour", "enchant_bow"}:
            self._enchant(action)
        elif action.startswith("level_up_"):
            self._level_up(action)
        else:
            self._append_action(action, "noop", {"noop": True})

        self._update_boss_logic()
        self._update_mobs(action)
        self._update_projectiles(action)
        self._spawn_mobs(action)
        self._update_plants()
        if bool(self.resolved.rules.get("homeostasis", False)):
            self._update_intrinsics(action)
        self._clip_inventory_and_intrinsics()
        self._calculate_inventory_achievements()
        self._apply_health_reward(before_health)
        self.world.light_level = _calculate_light_level(self.world.timestep, self._day_length())
        done_reason = self._done_reason()
        if done_reason is not None:
            self.private.terminated = done_reason in {"death", "boss_defeated"}
            self.private.truncated = done_reason == "max_steps"
            self.private.done_reason = done_reason
        self.public = self._public_state(done=self.private.terminated or self.private.truncated)
        self._append_inventory_deltas(action, before_inventory, self.world.inventory)
        for achievement in self._newly_unlocked(before_achievements):
            self.private.achievements.add(achievement)
            self._append_achievement(action, achievement)
        if before_pos != self.world.player_pos or before_level != self.world.player_level:
            self.nev.append(
                step_index=self.private.step_index,
                episode_id=self.private.episode_id,
                kind="state_transition",
                action=action,
                transition="pose",
                message=f"StateTransition(level={self.world.player_level},pos={list(self.world.player_pos)})",
                payload={"from": {"level": before_level, "pos": list(before_pos), "tile": before_tile}, "to": {"level": self.world.player_level, "pos": list(self.world.player_pos), "tile": self.tile_at(self.world.player_pos)}},
            )
        if self.private.reward_last != 0.0:
            self.nev.append(
                step_index=self.private.step_index,
                episode_id=self.private.episode_id,
                kind="reward_delta",
                action=action,
                transition="reward",
                message=f"RewardDelta({self.private.reward_last:.2f},total={self.private.total_reward:.2f})",
                payload={"reward": self.private.reward_last, "total_reward": self.private.total_reward},
            )
        if self.private.terminated or self.private.truncated:
            if self.private.done_reason == "death":
                kind = "death"
            elif self.private.done_reason == "max_steps":
                kind = "episode_truncated"
            else:
                kind = "terminal_success"
            self.nev.append(
                step_index=self.private.step_index,
                episode_id=self.private.episode_id,
                kind=kind,
                action=action,
                transition=self.private.done_reason,
                message=f"{_title_event(kind)}({self.private.done_reason})",
                payload={"reason": self.private.done_reason},
            )
            self.nev.append(
                step_index=self.private.step_index,
                episode_id=self.private.episode_id,
                kind="terminal",
                action=action,
                transition=self.private.done_reason,
                message=f"Terminal({self.private.done_reason})",
                payload={"reason": self.private.done_reason},
            )
        self._checkpoint_cadence_event()
        self._assert_runtime_invariants(f"step:{action}")
        return self.symbolic_readout()

    def checkpoint_bytes(self) -> bytes:
        if self.world is None or self.resolved is None:
            raise RuntimeError("engine must be reset before checkpoint")
        self.world.rng_state = self._rng.getstate()
        self._assert_runtime_invariants("checkpoint")
        return encode_checkpoint(
            env_family=self.ENV_FAMILY,
            episode_id=self.private.episode_id,
            step_index=self.private.step_index,
            nev_cursor=self.nev.cursor(),
            config_hash=self.private.config_hash,
            sim={
                "resolved": self.resolved.to_dict(),
                "world": self.world.to_checkpoint(),
                "private": self.private.to_dict(),
            },
            nev_events=self.nev.export(),
        )

    def restore_checkpoint(self, blob: bytes) -> int:
        payload = decode_checkpoint(blob)
        if payload.get("env_family") != self.ENV_FAMILY:
            raise ValueError(f"wrong env_family: {payload.get('env_family')}")
        sim = payload["sim"]
        self.resolved = _resolved_from_dict(sim["resolved"])
        self.world = CraftaxWorld.from_checkpoint(sim["world"])
        self._rng = random.Random(self.world.seed)
        if self.world.rng_state is None:
            raise CraftaxInvariantError("checkpoint missing rng_state; deterministic restore would diverge")
        try:
            self._rng.setstate(_json_to_rng_state(self.world.rng_state))
        except (TypeError, ValueError) as exc:
            raise CraftaxInvariantError(f"invalid checkpoint rng_state: {exc}") from exc
        private = sim["private"]
        self.private = PrivateState(
            episode_id=str(private["episode_id"]),
            task_id=str(private["task_id"]),
            scenario_id=str(private.get("scenario_id", private["task_id"])),
            seed=_strict_int(private["seed"], "private.seed"),
            config_hash=str(private["config_hash"]),
            step_index=_strict_int(private["step_index"], "private.step_index"),
            reward_last=_strict_finite_number(private.get("reward_last", 0.0), "private.reward_last"),
            total_reward=_strict_finite_number(private.get("total_reward", 0.0), "private.total_reward"),
            terminated=_strict_bool(private.get("terminated", False), "private.terminated"),
            truncated=_strict_bool(private.get("truncated", False), "private.truncated"),
            done_reason=private.get("done_reason"),
            achievements=set(private.get("achievements", [])),
            invalid_action_count=_strict_int(private.get("invalid_action_count", 0), "private.invalid_action_count"),
        )
        self._assert_checkpoint_envelope(payload)
        self._assert_runtime_invariants("restore")
        self.public = self._public_state(done=self.private.terminated or self.private.truncated)
        events = self._checkpoint_events_from_payload(payload)
        self._assert_checkpoint_reward_state(events)
        self.nev = NevLog.from_export(events)
        return self.nev.cursor()

    def _assert_checkpoint_envelope(self, payload: dict[str, Any]) -> None:
        if self.private is None:
            return
        envelope_episode = payload.get("episode_id")
        if not isinstance(envelope_episode, str) or envelope_episode != self.private.episode_id:
            raise CraftaxInvariantError(f"checkpoint episode_id={envelope_episode!r} private.episode_id={self.private.episode_id!r}")
        envelope_config = payload.get("config_hash")
        if not isinstance(envelope_config, str) or envelope_config != self.private.config_hash:
            raise CraftaxInvariantError(f"checkpoint config_hash={envelope_config!r} private.config_hash={self.private.config_hash!r}")
        envelope_step = payload.get("step_index")
        if not isinstance(envelope_step, int) or isinstance(envelope_step, bool) or envelope_step != self.private.step_index:
            raise CraftaxInvariantError(f"checkpoint step_index={envelope_step!r} private.step_index={self.private.step_index!r}")

    def _checkpoint_events_from_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        if self.private is None:
            return []
        cursor = _strict_int(payload.get("nev_cursor"), "checkpoint.nev_cursor")
        if cursor < 0:
            raise CraftaxInvariantError(f"checkpoint.nev_cursor must be non-negative: {cursor!r}")

        has_full_events = "nev_events" in payload
        if has_full_events:
            events = payload["nev_events"]
            if not isinstance(events, list):
                raise CraftaxInvariantError(f"checkpoint.nev_events must be a list: {type(events).__name__}")
            if cursor != len(events):
                raise CraftaxInvariantError(f"checkpoint.nev_cursor={cursor} event_count={len(events)}")
            digest = payload.get("nev_event_digest")
            if not isinstance(digest, str) or digest != digest_json(events):
                raise CraftaxInvariantError(f"checkpoint.nev_event_digest mismatch: {digest!r}")
            self._assert_checkpoint_tail(payload, cursor, events)
            self._assert_checkpoint_events(events)
            return events

        sim_events = payload.get("sim", {}).get("events")
        if sim_events is not None:
            if not isinstance(sim_events, list):
                raise CraftaxInvariantError(f"checkpoint.sim.events must be a list: {type(sim_events).__name__}")
            if cursor != len(sim_events):
                raise CraftaxInvariantError(f"checkpoint.nev_cursor={cursor} sim_event_count={len(sim_events)}")
            self._assert_checkpoint_events(sim_events)
            return sim_events

        tail_events = payload.get("nev_tail_events")
        if tail_events is None:
            if cursor != 0:
                raise CraftaxInvariantError(f"checkpoint.nev_cursor={cursor} but no events are present")
            return []
        if not isinstance(tail_events, list):
            raise CraftaxInvariantError(f"checkpoint.nev_tail_events must be a list: {type(tail_events).__name__}")
        tail_offset = _strict_int(payload.get("nev_tail_cursor_offset"), "checkpoint.nev_tail_cursor_offset")
        if tail_offset != 0:
            raise CraftaxInvariantError(f"tail-only checkpoint cannot preserve nonzero NEV cursor offset: {tail_offset}")
        if tail_offset < 0 or tail_offset + len(tail_events) != cursor:
            raise CraftaxInvariantError(f"checkpoint tail cursor mismatch: offset={tail_offset} tail_count={len(tail_events)} cursor={cursor}")
        self._assert_checkpoint_events(tail_events)
        return tail_events

    def _assert_checkpoint_tail(self, payload: dict[str, Any], cursor: int, events: list[dict[str, Any]]) -> None:
        tail_events = payload.get("nev_tail_events")
        if tail_events is None:
            return
        if not isinstance(tail_events, list):
            raise CraftaxInvariantError(f"checkpoint.nev_tail_events must be a list: {type(tail_events).__name__}")
        tail_offset = _strict_int(payload.get("nev_tail_cursor_offset"), "checkpoint.nev_tail_cursor_offset")
        expected_tail = events[-NEV_TAIL_EVENTS:]
        expected_offset = max(0, cursor - len(expected_tail))
        if tail_offset != expected_offset:
            raise CraftaxInvariantError(f"checkpoint.nev_tail_cursor_offset={tail_offset} expected={expected_offset}")
        if tail_events != expected_tail:
            raise CraftaxInvariantError("checkpoint.nev_tail_events do not match full event suffix")

    def _assert_checkpoint_events(self, events: list[dict[str, Any]]) -> None:
        if self.private is None:
            return
        previous_step = -1
        for index, event in enumerate(events):
            if not isinstance(event, dict):
                raise CraftaxInvariantError(f"checkpoint event[{index}] must be an object: {event!r}")
            step = _strict_int(event.get("step_index"), f"checkpoint event[{index}].step_index")
            tick = _strict_int(event.get("tick", step), f"checkpoint event[{index}].tick")
            if tick != step:
                raise CraftaxInvariantError(f"checkpoint event[{index}] tick={tick} step_index={step}")
            if step < previous_step:
                raise CraftaxInvariantError(f"checkpoint event[{index}] step_index={step} before previous={previous_step}")
            if step > self.private.step_index:
                raise CraftaxInvariantError(f"checkpoint event[{index}] step_index={step} private.step_index={self.private.step_index}")
            previous_step = step
            episode_id = event.get("episode_id")
            if not isinstance(episode_id, str) or episode_id != self.private.episode_id:
                raise CraftaxInvariantError(f"checkpoint event[{index}] episode_id={episode_id!r} private.episode_id={self.private.episode_id!r}")
            for field in ("kind", "message"):
                value = event.get(field)
                if not isinstance(value, str) or not value:
                    raise CraftaxInvariantError(f"checkpoint event[{index}].{field} must be a non-empty string: {value!r}")
            kind = str(event["kind"])
            if kind not in VALID_NEV_EVENT_KINDS:
                raise CraftaxInvariantError(f"checkpoint event[{index}].kind is not emitted by Craftax runtime: {kind!r}")
            severity = event.get("severity", "info")
            if severity not in {"info", "warn"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}].severity must be info or warn: {severity!r}")
            payload = event.get("payload", {})
            if not isinstance(payload, dict):
                raise CraftaxInvariantError(f"checkpoint event[{index}].payload must be an object: {payload!r}")
            action = event.get("action")
            if action is not None:
                if isinstance(action, str):
                    if not action:
                        raise CraftaxInvariantError(f"checkpoint event[{index}].action must be non-empty when present")
                elif not isinstance(action, dict):
                    raise CraftaxInvariantError(f"checkpoint event[{index}].action must be a string, object, or null: {action!r}")
            transition = event.get("transition")
            if transition is not None and (not isinstance(transition, str) or not transition):
                raise CraftaxInvariantError(f"checkpoint event[{index}].transition must be a non-empty string or null: {transition!r}")
            self._assert_checkpoint_event_payload(index, event)

    def _assert_checkpoint_event_payload(self, index: int, event: dict[str, Any]) -> None:
        kind = str(event["kind"])
        payload = event.get("payload", {})
        transition = event.get("transition")
        step = _strict_int(event.get("step_index"), f"checkpoint event[{index}].step_index")

        if kind == "achievement_unlocked":
            achievement = payload.get("achievement")
            if achievement not in CRAFTAX_ACHIEVEMENTS:
                raise CraftaxInvariantError(f"checkpoint event[{index}].payload.achievement is unsupported: {achievement!r}")
            if transition != achievement:
                raise CraftaxInvariantError(f"checkpoint event[{index}] achievement transition={transition!r} payload={achievement!r}")
            return

        if kind == "resource_delta":
            if transition not in default_inventory():
                raise CraftaxInvariantError(f"checkpoint event[{index}] resource transition is unsupported: {transition!r}")
            resource = payload.get("resource")
            if resource is not None and resource != transition:
                raise CraftaxInvariantError(f"checkpoint event[{index}] resource={resource!r} transition={transition!r}")
            if "delta" in payload and payload.get("delta") is not None:
                before = payload.get("before")
                after = payload.get("after")
                delta = payload.get("delta")
                if not (_finite_number(before) and _finite_number(after) and _finite_number(delta)):
                    raise CraftaxInvariantError(f"checkpoint event[{index}] numeric resource delta fields must be finite")
                if abs((float(after) - float(before)) - float(delta)) > 1e-9:
                    raise CraftaxInvariantError(f"checkpoint event[{index}] resource delta does not match before/after")
            return

        if kind == "rule_violation":
            code = payload.get("code")
            action = payload.get("action")
            if event.get("severity") != "warn" or transition != "reject":
                raise CraftaxInvariantError(f"checkpoint event[{index}] rule_violation must be warn/reject")
            if code not in {"terminal", "unknown_action"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}] unsupported rule_violation code: {code!r}")
            if action != event.get("action"):
                raise CraftaxInvariantError(f"checkpoint event[{index}] rule_violation action payload mismatch")
            return

        if kind in {"death", "episode_truncated", "terminal", "terminal_success"}:
            reason = payload.get("reason")
            if reason not in {"death", "boss_defeated", "max_steps"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}] unsupported terminal reason: {reason!r}")
            if transition != reason:
                raise CraftaxInvariantError(f"checkpoint event[{index}] terminal transition={transition!r} reason={reason!r}")
            if kind == "episode_truncated" and reason != "max_steps":
                raise CraftaxInvariantError(f"checkpoint event[{index}] episode_truncated reason must be max_steps")
            if kind == "terminal_success" and reason != "boss_defeated":
                raise CraftaxInvariantError(f"checkpoint event[{index}] terminal_success reason must be boss_defeated")
            if kind == "death" and reason != "death":
                raise CraftaxInvariantError(f"checkpoint event[{index}] death reason must be death")
            return

        if kind == "checkpoint_cadence":
            cadence_step = _strict_int(payload.get("step_index"), f"checkpoint event[{index}].payload.step_index")
            nev_cursor = _strict_int(payload.get("nev_cursor"), f"checkpoint event[{index}].payload.nev_cursor")
            if cadence_step != step or nev_cursor < 0 or transition != "cadence":
                raise CraftaxInvariantError(f"checkpoint event[{index}] invalid checkpoint_cadence payload")
            return

        if kind == "floor_transition":
            if transition not in {"ascend", "descend"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}] invalid floor transition: {transition!r}")
            level = _strict_int(payload.get("level"), f"checkpoint event[{index}].payload.level")
            if level < 0:
                raise CraftaxInvariantError(f"checkpoint event[{index}] floor level must be non-negative: {level!r}")
            return

        if kind == "state_transition":
            if transition != "pose":
                raise CraftaxInvariantError(f"checkpoint event[{index}] state_transition must use pose transition")
            self._assert_checkpoint_pose_payload(index, payload.get("from"), "from")
            self._assert_checkpoint_pose_payload(index, payload.get("to"), "to")
            return

        if kind == "entity_transition":
            self._assert_checkpoint_entity_payload(index, payload.get("entity"), "entity")
            if transition not in {"attack_player", "damage", "defeat", "despawn", "eat_passive", "ranged_attack", "spawn"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}] unsupported entity transition: {transition!r}")
            return

        if kind == "combat":
            if transition != "mob_attack":
                raise CraftaxInvariantError(f"checkpoint event[{index}] combat transition must be mob_attack")
            self._assert_checkpoint_entity_payload(index, payload.get("entity"), "entity")
            if not _finite_number(payload.get("damage")) or float(payload["damage"]) < 0.0:
                raise CraftaxInvariantError(f"checkpoint event[{index}] combat damage must be non-negative finite")
            return

        if kind == "projectile_transition":
            self._assert_checkpoint_projectile_payload(index, payload.get("projectile"), "projectile")
            if transition not in {"despawn", "hit_mob", "hit_player", "move", "spawn"}:
                raise CraftaxInvariantError(f"checkpoint event[{index}] unsupported projectile transition: {transition!r}")
            for field in ("pos", "from", "to", "hit_pos", "direction"):
                if field in payload:
                    _pos(payload[field], f"checkpoint event[{index}].payload.{field}")
            if "entity" in payload:
                self._assert_checkpoint_entity_payload(index, payload["entity"], "entity")
            if "source" in payload:
                self._assert_checkpoint_entity_payload(index, payload["source"], "source")
            if "damage" in payload and (not _finite_number(payload["damage"]) or float(payload["damage"]) < 0.0):
                raise CraftaxInvariantError(f"checkpoint event[{index}] projectile damage must be non-negative finite")
            return

        if kind == "action_applied":
            for field in ("target", "to", "pos"):
                if field in payload:
                    value = payload[field]
                    if isinstance(value, str) and field == "target":
                        if not value:
                            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field} must be non-empty")
                    else:
                        _pos(value, f"checkpoint event[{index}].payload.{field}")
            if "entity" in payload:
                self._assert_checkpoint_entity_payload(index, payload["entity"], "entity")
            return

        if kind == "task_resolved":
            resolved = payload.get("resolved")
            if not isinstance(resolved, dict):
                raise CraftaxInvariantError(f"checkpoint event[{index}].payload.resolved must be an object")
            if resolved.get("episode_id") != self.private.episode_id or resolved.get("config_hash") != self.private.config_hash:
                raise CraftaxInvariantError(f"checkpoint event[{index}] task_resolved payload disagrees with restored identity")

    def _assert_checkpoint_pose_payload(self, index: int, value: Any, field: str) -> None:
        if not isinstance(value, dict):
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field} must be an object")
        level = _strict_int(value.get("level"), f"checkpoint event[{index}].payload.{field}.level")
        if level < 0:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.level must be non-negative")
        _pos(value.get("pos"), f"checkpoint event[{index}].payload.{field}.pos")
        tile = value.get("tile")
        if not isinstance(tile, str) or tile not in VALID_POSE_TILES:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.tile is unsupported: {tile!r}")

    def _assert_checkpoint_entity_payload(self, index: int, value: Any, field: str) -> None:
        if not isinstance(value, dict):
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field} must be an object")
        kind = value.get("kind")
        if kind not in MOB_NAMES:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.kind is unsupported: {kind!r}")
        mob_class = value.get("class", value.get("mob_class", _mob_class(str(kind))))
        if mob_class not in {"passive", "melee", "ranged"}:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.class is unsupported: {mob_class!r}")
        level = _strict_int(value.get("level", 0), f"checkpoint event[{index}].payload.{field}.level")
        if level < 0:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.level must be non-negative")
        _pos(value.get("pos"), f"checkpoint event[{index}].payload.{field}.pos")
        health = value.get("health")
        if not _finite_number(health):
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.health must be finite")
        if "mask" in value:
            _strict_bool(value["mask"], f"checkpoint event[{index}].payload.{field}.mask")
        if "attack_cooldown" in value:
            _strict_int(value["attack_cooldown"], f"checkpoint event[{index}].payload.{field}.attack_cooldown")

    def _assert_checkpoint_projectile_payload(self, index: int, value: Any, field: str) -> None:
        if not isinstance(value, dict):
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field} must be an object")
        kind = value.get("kind")
        if kind not in PROJECTILE_TYPES:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.kind is unsupported: {kind!r}")
        owner = value.get("owner")
        if owner not in VALID_PROJECTILE_OWNERS:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.owner is unsupported: {owner!r}")
        if not _projectile_kind_allowed_for_owner(str(kind), str(owner)):
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.kind={kind!r} cannot be owned by {owner!r}")
        level = _strict_int(value.get("level", 0), f"checkpoint event[{index}].payload.{field}.level")
        if level < 0:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.level must be non-negative")
        _pos(value.get("pos"), f"checkpoint event[{index}].payload.{field}.pos")
        direction = _dir(value.get("direction"), f"checkpoint event[{index}].payload.{field}.direction")
        if direction not in VALID_DIRECTIONS:
            raise CraftaxInvariantError(f"checkpoint event[{index}].payload.{field}.direction must be cardinal: {direction!r}")
        if "mask" in value:
            _strict_bool(value["mask"], f"checkpoint event[{index}].payload.{field}.mask")

    def _assert_checkpoint_reward_state(self, events: list[dict[str, Any]]) -> None:
        if self.private is None:
            return
        reward_steps: set[int] = set()
        running_total = 0.0
        current_step_reward = 0.0
        for index, event in enumerate(events):
            if event.get("kind") != "reward_delta":
                continue
            step = _strict_int(event.get("step_index"), f"checkpoint reward_delta[{index}].step_index")
            if step in reward_steps:
                raise CraftaxInvariantError(f"checkpoint has duplicate reward_delta events for step {step}")
            reward_steps.add(step)
            payload = event.get("payload")
            if not isinstance(payload, dict):
                raise CraftaxInvariantError(f"checkpoint reward_delta[{index}].payload must be an object: {payload!r}")
            reward = _strict_finite_number(payload.get("reward"), f"checkpoint reward_delta[{index}].payload.reward")
            total_reward = _strict_finite_number(payload.get("total_reward"), f"checkpoint reward_delta[{index}].payload.total_reward")
            running_total += reward
            if abs(total_reward - running_total) > 1e-9:
                raise CraftaxInvariantError(
                    f"checkpoint reward_delta[{index}] total_reward={total_reward!r} expected cumulative {running_total!r}"
                )
            if step == self.private.step_index:
                current_step_reward = reward
        if abs(self.private.reward_last - current_step_reward) > 1e-9:
            raise CraftaxInvariantError(
                f"private.reward_last={self.private.reward_last!r} checkpoint current-step reward={current_step_reward!r}"
            )
        if abs(self.private.total_reward - running_total) > 1e-9:
            raise CraftaxInvariantError(
                f"private.total_reward={self.private.total_reward!r} checkpoint reward history total={running_total!r}"
            )

    def clone_for_sim(self) -> "CraftaxEngine":
        clone = CraftaxEngine()
        clone.restore_checkpoint(self.checkpoint_bytes())
        return clone

    def simulate(self, sequences: list[list[str]]) -> list[dict[str, Any]]:
        root = self.checkpoint_bytes()
        results: list[dict[str, Any]] = []
        for index, sequence in enumerate(sequences):
            sim = CraftaxEngine()
            sim.restore_checkpoint(root)
            rewards: list[float] = []
            for action in sequence:
                if sim.private.terminated or sim.private.truncated:
                    break
                sim.step(action)
                rewards.append(sim.private.reward_last)
            results.append(
                {
                    "index": index,
                    "actions": list(sequence),
                    "reward": sim.private.total_reward,
                    "reward_trace": rewards,
                    "achievements": sorted(sim.private.achievements),
                    "terminated": sim.private.terminated,
                    "truncated": sim.private.truncated,
                    "steps": sim.private.step_index,
                    "readout": sim.symbolic_readout(),
                }
            )
        return results

    def valid_actions(self) -> list[str]:
        if self.private.terminated or self.private.truncated:
            return []
        return list(ACTION_NAMES)

    def symbolic_readout(self) -> dict[str, Any]:
        if self.world is None or self.resolved is None:
            raise RuntimeError("engine must be reset before readout")
        observation = self._observation()
        return {
            "schema": "gamebench.craftax.readout.v1",
            "env_family": self.ENV_FAMILY,
            "task_id": self.resolved.task_id,
            "scenario_id": self.resolved.scenario_id,
            "public": self.public.to_dict(),
            "private": self.private.to_dict(),
            "observation": observation,
            "observation_text": observation_text_from_readout(observation, self.valid_actions()),
            "ascii": self._ascii_map(),
            "grid_hash": _stable_hash({
                "level": self.world.player_level,
                "map": self.world.maps[self.world.player_level],
                "item_map": self.world.item_maps[self.world.player_level],
                "monsters_killed": self.world.monsters_killed,
                "entities": [e.to_dict() for e in self.world.entities if e.mask],
                "player_projectiles": [p.to_dict() for p in self.world.player_projectiles if p.mask],
                "mob_projectiles": [p.to_dict() for p in self.world.mob_projectiles if p.mask],
            }),
            "nev_cursor": self.nev.cursor(),
            "valid_actions": self.valid_actions(),
        }

    def _make_world(self, resolved: ResolvedTask) -> CraftaxWorld:
        width, height, levels = resolved.width, resolved.height, int(resolved.world.get("levels", 9))
        layout = generate_world_layout(width, height, levels, self._rng, resolved.world.get("densities", {}))
        monsters_killed = [0 for _ in range(levels)]
        if monsters_killed:
            monsters_killed[0] = 10
        potion_mapping = list(range(6))
        self._rng.shuffle(potion_mapping)
        world = CraftaxWorld(
            width=width,
            height=height,
            levels=levels,
            max_steps=resolved.max_steps,
            seed=resolved.seed,
            maps=layout.maps,
            item_maps=layout.item_maps,
            light_maps=layout.light_maps,
            down_ladders=layout.down_ladders,
            up_ladders=layout.up_ladders,
            chests_opened=[False for _ in range(levels)],
            monsters_killed=monsters_killed,
            potion_mapping=potion_mapping,
            player_pos=layout.player_pos,
            light_level=_calculate_light_level(0, _day_length_from_resolved(resolved)),
        )
        world.entities = self._spawn_entities(world)
        self._apply_initial_state(world, resolved.world.get("initial_state"))
        return world

    def _spawn_entities(self, world: CraftaxWorld) -> list[Entity]:
        entities: list[Entity] = []
        occupied: set[tuple[int, tuple[int, int]]] = set()
        per_level = max(2, world.width // 8)
        idx = 0
        for level in range(world.levels):
            if level == 8:
                continue
            choices = FLOOR_MOBS.get(level, FLOOR_MOBS[0])
            for local_idx in range(per_level):
                kind = choices[local_idx % len(choices)]
                mob_class = _mob_class(kind)
                for _ in range(80):
                    pos = (self._rng.randrange(1, world.width - 1), self._rng.randrange(1, world.height - 1))
                    block = normalize_tile(world.maps[level][pos[1]][pos[0]])
                    if self._mob_can_occupy_block(kind, mob_class, block) and (level, pos) not in occupied and (level != world.player_level or pos != world.player_pos):
                        entities.append(Entity(id=f"{kind}_{idx}", kind=kind, pos=pos, health=float(MOB_HEALTH[kind]), level=level, mob_class=mob_class))
                        occupied.add((level, pos))
                        idx += 1
                        break
        return entities

    def _apply_initial_state(self, world: CraftaxWorld, initial: dict[str, Any] | None) -> None:
        if not initial:
            return
        if "player" in initial:
            player = initial["player"]
            world.player_pos = _pos(player.get("pos", world.player_pos))
            world.player_direction = _dir(player.get("direction", player.get("facing", world.player_direction)))
            world.player_level = _strict_int(player.get("level", world.player_level), "player.level")
        for patch in initial.get("tiles", []):
            level = _strict_int(patch.get("level", world.player_level), "tile.level")
            x, y = _pos(patch["pos"])
            self._set_cell(world, level, (x, y), str(patch["kind"]))
        for patch in initial.get("items", []):
            level = _strict_int(patch.get("level", world.player_level), "item.level")
            x, y = _pos(patch["pos"])
            self._set_item_cell(world, level, (x, y), str(patch["kind"]))
        if "inventory" in initial:
            world.inventory = normalize_inventory(_deep_merge(default_inventory(), initial["inventory"]))
        for name, count in (initial.get("achievements") or {}).items():
            world.achievements[str(name)] = _strict_int(count, f"achievements[{name}]")
        if "monsters_killed" in initial:
            world.monsters_killed = _normalize_int_vector(initial["monsters_killed"], world.levels, "monsters_killed")
        if "chests_opened" in initial:
            world.chests_opened = _normalize_bool_vector(initial["chests_opened"], world.levels, "chests_opened")
        if "potion_mapping" in initial:
            world.potion_mapping = _normalize_int_vector(initial["potion_mapping"], 6, "potion_mapping")
        if "intrinsics" in initial:
            intrinsics = initial["intrinsics"]
            world.is_sleeping = _strict_bool(intrinsics.get("is_sleeping", world.is_sleeping), "intrinsics.is_sleeping")
            world.is_resting = _strict_bool(intrinsics.get("is_resting", world.is_resting), "intrinsics.is_resting")
            world.player_recover = _strict_finite_number(intrinsics.get("recover", world.player_recover), "intrinsics.recover")
            world.player_hunger = _strict_finite_number(intrinsics.get("hunger", world.player_hunger), "intrinsics.hunger")
            world.player_thirst = _strict_finite_number(intrinsics.get("thirst", world.player_thirst), "intrinsics.thirst")
            world.player_fatigue = _strict_finite_number(intrinsics.get("fatigue", world.player_fatigue), "intrinsics.fatigue")
            world.player_recover_mana = _strict_finite_number(intrinsics.get("recover_mana", world.player_recover_mana), "intrinsics.recover_mana")
        if "growing_plants" in initial:
            world.growing_plants = copy.deepcopy(initial["growing_plants"])
        if "entities" in initial:
            world.entities = []
            for idx, item in enumerate(initial["entities"]):
                kind = normalize_mob(str(item["kind"]))
                level = _strict_int(item.get("level", world.player_level), "entity.level")
                x, y = _pos(item["pos"], "entity.pos")
                world.entities.append(
                    Entity(
                        id=str(item.get("id", f"{kind}_{idx}")),
                        kind=kind,
                        pos=(x, y),
                        health=float(item.get("health", MOB_HEALTH.get(kind, 3))),
                        level=level,
                        mob_class=str(item.get("class") or item.get("mob_class") or _mob_class(kind)),
                        attack_cooldown=_strict_int(item.get("attack_cooldown", 0), "entity.attack_cooldown"),
                        mask=_strict_bool(item.get("mask", True), "entity.mask"),
                    )
                )
        if "player_projectiles" in initial:
            world.player_projectiles = _projectiles_from_initial(initial["player_projectiles"], "player", world.player_level)
        if "mob_projectiles" in initial:
            world.mob_projectiles = _projectiles_from_initial(initial["mob_projectiles"], "mob", world.player_level)

    def _move(self, action: str) -> None:
        assert self.world is not None
        dx, dy = DIRS[action]
        self.world.player_direction = (dx, dy)
        target = (self.world.player_pos[0] + dx, self.world.player_pos[1] + dy)
        entity = self.entity_at(target, self.world.player_level)
        if entity is not None:
            self._append_action(action, "blocked_by_mob", {"target": list(target), "entity": entity.to_dict()})
            return
        if not self.in_bounds(target):
            self._append_action(action, "blocked", {"reason": "out_of_bounds", "target": list(target)})
            return
        tile = self.block_at(target)
        if _is_land_walkable(tile):
            self.world.player_pos = target
            self._append_action(action, "move", {"to": list(target), "tile": self.tile_at(target), "block": tile, "item": self.item_at(target)})
            return
        self._append_action(action, "blocked", {"reason": f"blocked:{tile}", "target": list(target), "tile": tile})

    def _weighted_choice(self, weighted_items: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
        total = sum(float(item[1]) for item in weighted_items)
        roll = self._rng.random() * total
        running = 0.0
        for item in weighted_items:
            running += float(item[1])
            if roll < running:
                return item
        return weighted_items[-1]

    def _loot_chest(self) -> None:
        assert self.world is not None
        inv = self.world.inventory
        level = self.world.player_level
        first_chest_on_level = not bool(self.world.chests_opened[level])

        looting_wood = self._rng.random() < 0.6
        wood_amount = self._rng.randrange(1, 6)
        if looting_wood:
            inv["wood"] += wood_amount

        looting_torch = self._rng.random() < 0.6
        torch_amount = self._rng.randrange(4, 8)
        if looting_torch:
            inv["torches"] += torch_amount

        looting_ore = self._rng.random() < 0.6
        ore, _, min_amount, max_amount = self._weighted_choice(CHEST_ORE_LOOT)
        ore_amount = self._rng.randrange(int(min_amount), int(max_amount))
        if looting_ore:
            inv[str(ore)] += ore_amount

        looting_potion = self._rng.random() < 0.5
        potion_color = POTION_COLORS[self._rng.randrange(0, 6)]
        potion_amount = self._rng.randrange(1, 3)
        if looting_potion:
            inv["potions"][potion_color] += potion_amount

        looting_arrows = self._rng.random() < 0.25
        arrows_amount = self._rng.randrange(1, 5)
        if looting_arrows:
            inv["arrows"] += arrows_amount

        looting_tool = self._rng.random() < 0.2
        tool_id = self._rng.randrange(0, 2)
        pickaxe_level = int(self._weighted_choice(CHEST_TOOL_LOOT)[0])
        sword_level = int(self._weighted_choice(CHEST_TOOL_LOOT)[0])
        if looting_tool and tool_id == 0:
            inv["pickaxe"] = max(int(inv["pickaxe"]), pickaxe_level)
        if looting_tool and tool_id == 1:
            inv["sword"] = max(int(inv["sword"]), sword_level)

        if level == 1 and first_chest_on_level:
            inv["bow"] = 1
        if level in {3, 4} and first_chest_on_level:
            inv["books"] += 1

    def _do(self) -> None:
        assert self.world is not None
        target = self.front_pos()
        entity = self.entity_at(target, self.world.player_level)
        if entity is not None:
            self._melee(entity, "do")
            return
        if not self.in_bounds(target):
            self._append_noop("do", "out_of_bounds", {"target": list(target)})
            return
        tile = self.block_at(target)
        if tile in RESOURCE_TILE:
            resource, achievement, required_pickaxe = RESOURCE_TILE[tile]
            if int(self.world.inventory["pickaxe"]) < required_pickaxe:
                self._append_noop("do", f"needs_pickaxe:{TIER_NAMES[required_pickaxe]}", {"target": list(target), "tile": tile})
                return
            self.world.inventory[resource] += 1
            self._set_tile(target, _resource_replacement(tile, self.world.player_level))
            self._unlock(achievement)
            self._append_action("do", "harvest", {"tile": tile, "resource": resource, "target": list(target)})
            return
        if tile == "grass":
            if self._rng.random() < 0.1:
                self.world.inventory["sapling"] += 1
                self._unlock("collect_sapling")
                self._append_action("do", "collect_sapling", {"target": list(target)})
                return
            self._append_noop("do", "no_sapling", {"target": list(target)})
            return
        if tile in {"water", "fountain"}:
            self.world.inventory["drink"] = min(_max_stat(self.world, "drink"), int(self.world.inventory["drink"]) + 1)
            self._unlock("collect_drink")
            self._append_action("do", "drink", {"tile": tile, "target": list(target)})
            return
        if tile == "ripe_plant":
            self.world.inventory["food"] = min(_max_stat(self.world, "food"), int(self.world.inventory["food"]) + 4)
            self._set_tile(target, "plant")
            self._reset_growing_plant(target)
            self._unlock("eat_plant")
            self._append_action("do", "eat_plant", {"target": list(target)})
            return
        if tile == "chest":
            self._loot_chest()
            self._set_tile(target, "path")
            self.world.chests_opened[self.world.player_level] = True
            self._unlock("open_chest")
            self._append_action("do", "open_chest", {"target": list(target)})
            return
        if tile in {"crafting_table", "furnace"}:
            self._set_tile(target, "path")
            self._append_action("do", "mine_block", {"tile": tile, "target": list(target), "replacement": "path"})
            return
        if tile == "necromancer":
            if self._is_fighting_boss() and self._is_boss_vulnerable():
                self.world.inventory["boss_progress"] = int(self.world.inventory.get("boss_progress", 0)) + 1
                self.world.boss_timesteps_to_spawn_this_round = BOSS_FIGHT_SPAWN_TURNS
                self._unlock("damage_necromancer")
                self._append_action("do", "boss_damage", {"target": list(target), "boss_progress": self.world.inventory["boss_progress"]})
                return
            self._append_action("do", "boss_not_vulnerable", {"target": list(target)})
            return
        self._append_noop("do", f"nothing_to_do:{tile}", {"target": list(target), "tile": tile})

    def _recover(self, action: str) -> None:
        assert self.world is not None
        inv = self.world.inventory
        before_energy = int(inv["energy"])
        inv["energy"] = min(_max_stat(self.world, "energy"), int(inv["energy"]) + (3 if action == "sleep" else 1))
        inv["health"] = min(_max_stat(self.world, "health"), int(inv["health"]) + 1)
        inv["mana"] = min(_max_stat(self.world, "mana"), int(inv["mana"]) + 1)
        if action == "sleep" and before_energy < _max_stat(self.world, "energy") <= int(inv["energy"]):
            self._unlock("wake_up")
        self._append_action(action, "recover", {"energy": inv["energy"], "health": inv["health"], "mana": inv["mana"]})

    def _descend(self) -> None:
        assert self.world is not None
        if self.item_at(self.world.player_pos) != "ladder_down":
            self._append_noop("descend", "not_on_ladder_down", {"pos": list(self.world.player_pos), "item": self.item_at(self.world.player_pos)})
            return
        if not bool(self.resolved and self.resolved.rules.get("god_mode", False)):
            if int(self.world.monsters_killed[self.world.player_level]) < MONSTERS_KILLED_TO_CLEAR_LEVEL:
                self._append_noop("descend", "level_not_cleared", {"level": self.world.player_level, "monsters_killed": self.world.monsters_killed[self.world.player_level]})
                return
        if self.world.player_level + 1 >= self.world.levels:
            self._append_noop("descend", "lowest_level", {"level": self.world.player_level})
            return
        self.world.player_level += 1
        self.world.player_pos = _valid_ladder_pos(self.world.up_ladders[self.world.player_level]) or _find_item(self.world.item_maps[self.world.player_level], "ladder_up") or (2, 2)
        achievement = LEVEL_ACHIEVEMENTS.get(self.world.player_level)
        if achievement:
            had_achievement = self.world.achievements.get(achievement, 0) > 0
            self._unlock(achievement)
            if not had_achievement:
                self.world.inventory["xp"] = int(self.world.inventory.get("xp", 0)) + 1
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="floor_transition", action="descend", transition="descend", message=f"FloorTransition(descend,{self.world.player_level})", payload={"level": self.world.player_level})

    def _ascend(self) -> None:
        assert self.world is not None
        if self.item_at(self.world.player_pos) != "ladder_up":
            self._append_noop("ascend", "not_on_ladder_up", {"pos": list(self.world.player_pos), "item": self.item_at(self.world.player_pos)})
            return
        if self.world.player_level <= 0:
            self._append_noop("ascend", "top_level", {"level": self.world.player_level})
            return
        self.world.player_level -= 1
        self.world.player_pos = _valid_ladder_pos(self.world.down_ladders[self.world.player_level]) or _find_item(self.world.item_maps[self.world.player_level], "ladder_down") or (self.world.width - 3, self.world.height - 3)
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="floor_transition", action="ascend", transition="ascend", message=f"FloorTransition(ascend,{self.world.player_level})", payload={"level": self.world.player_level})

    def _place(self, action: str) -> None:
        assert self.world is not None
        target = self.front_pos()
        if not self.in_bounds(target):
            self._append_noop(action, "out_of_bounds", {"target": list(target)})
            return
        entity = self.entity_at(target, self.world.player_level)
        if entity is not None:
            self._append_action(action, "blocked_by_mob", {"target": list(target), "entity": entity.to_dict()})
            return
        target_tile = self.block_at(target)
        target_item = self.item_at(target)
        costs = {
            "place_stone": {"stone": 1},
            "place_table": {"wood": 2},
            "place_furnace": {"stone": 1},
            "place_plant": {"sapling": 1},
            "place_torch": {"torches": 1},
        }
        tiles = {"place_stone": "stone", "place_table": "crafting_table", "place_furnace": "furnace", "place_plant": "plant", "place_torch": "torch"}
        valid_target = target_tile in BASE_WALKABLE and target_item == "none"
        if action == "place_stone":
            valid_target = target_tile == "water" or (target_tile in BASE_WALKABLE and target_item == "none")
        elif action == "place_plant":
            valid_target = target_tile == "grass" and target_item == "none"
        elif action == "place_torch":
            valid_target = target_tile in CAN_PLACE_ITEM_ON and target_item == "none"
        if not valid_target:
            self._append_noop(action, f"target_not_placeable:{target_tile}", {"target": list(target), "tile": target_tile, "item": target_item})
            return
        if not self._pay(costs[action]):
            self._append_noop(action, "missing_resources", {"costs": costs[action]})
            return
        if action == "place_torch":
            self._set_item(target, "torch")
            self._add_torch_light(target)
        else:
            self._set_tile(target, tiles[action])
            if action == "place_plant":
                self._add_growing_plant(target)
        self._unlock(action)
        self._append_action(action, "place", {"tile": tiles[action], "target": list(target)})

    def _craft(self, action: str) -> None:
        assert self.world is not None
        if not self.near_tile({"crafting_table"}):
            self._append_noop(action, "needs_crafting_table")
            return
        recipes: dict[str, tuple[dict[str, int], str, int | None, int]] = {
            "make_wood_pickaxe": ({"wood": 1}, "pickaxe", 1, 1),
            "make_stone_pickaxe": ({"wood": 1, "stone": 1}, "pickaxe", 2, 1),
            "make_iron_pickaxe": ({"wood": 1, "stone": 1, "iron": 1, "coal": 1}, "pickaxe", 3, 1),
            "make_diamond_pickaxe": ({"wood": 1, "diamond": 3}, "pickaxe", 4, 1),
            "make_wood_sword": ({"wood": 1}, "sword", 1, 1),
            "make_stone_sword": ({"wood": 1, "stone": 1}, "sword", 2, 1),
            "make_iron_sword": ({"wood": 1, "stone": 1, "iron": 1, "coal": 1}, "sword", 3, 1),
            "make_diamond_sword": ({"wood": 1, "diamond": 2}, "sword", 4, 1),
            "make_arrow": ({"wood": 1, "stone": 1}, "arrows", None, 2),
            "make_torch": ({"wood": 1, "coal": 1}, "torches", None, 4),
            "make_iron_armour": ({"iron": 3, "coal": 3}, "armour", 1, 1),
            "make_diamond_armour": ({"diamond": 3}, "armour", 2, 1),
        }
        costs, target, tier, amount = recipes[action]
        if action.startswith("make_iron_") and not self.near_tile({"furnace"}):
            self._append_noop(action, "needs_furnace")
            return
        if action == "make_iron_armour" and not any(piece < 1 for piece in _armour(self.world)):
            self._append_noop(action, "armour_already_iron", {"armour": _armour(self.world)})
            return
        if action == "make_diamond_armour" and not any(piece < 2 for piece in _armour(self.world)):
            self._append_noop(action, "armour_already_diamond", {"armour": _armour(self.world)})
            return
        if target in {"pickaxe", "sword"} and tier is not None and int(self.world.inventory[target]) >= tier:
            self._append_noop(action, "already_have_tier", {"target": target, "tier": tier, "inventory_value": self.world.inventory[target]})
            return
        if target in {"arrows", "torches"} and int(self.world.inventory[target]) >= 99:
            self._append_noop(action, "inventory_cap", {"target": target, "inventory_value": self.world.inventory[target]})
            return
        if not self._pay(costs):
            self._append_noop(action, "missing_resources", {"costs": costs})
            return
        if target in {"pickaxe", "sword"}:
            self.world.inventory[target] = max(int(self.world.inventory[target]), int(tier or 0))
        elif target == "armour":
            armour = _armour(self.world)
            desired = int(tier or 0)
            for index, piece in enumerate(armour):
                if piece < desired:
                    armour[index] = desired
                    break
            self.world.inventory["armour"] = armour
        elif target == "arrows":
            self.world.inventory[target] += amount
        else:
            self.world.inventory[target] += amount
        self._unlock(action)
        self._append_action(action, "craft", {"target": target, "inventory_value": self.world.inventory[target]})

    def _shoot_arrow(self) -> None:
        assert self.world is not None
        if int(self.world.inventory["bow"]) <= 0 or int(self.world.inventory["arrows"]) <= 0:
            self._append_action("shoot_arrow", "noop", {"reason": "needs_bow_and_arrow"})
            return
        if self._spawn_player_projectile("arrow2", "shoot_arrow"):
            self.world.inventory["arrows"] -= 1
            self._unlock("fire_bow")
            self._append_action("shoot_arrow", "projectile_spawn", {"projectile": "arrow2"})

    def _cast_spell(self) -> None:
        assert self.world is not None
        learned_spells = self.world.inventory["learned_spells"]
        if not learned_spells or int(self.world.inventory["mana"]) < 2:
            self._append_action("cast_spell", "noop", {"reason": "spell_not_ready:fireball"})
            return
        if self._spawn_player_projectile("fireball", "cast_spell"):
            self.world.inventory["mana"] -= 2
            self._unlock("cast_spell")
            self._append_action("cast_spell", "projectile_spawn", {"projectile": "fireball"})

    def _drink_potion(self, color: str) -> None:
        assert self.world is not None
        if int(self.world.inventory["potions"].get(color, 0)) <= 0:
            self._append_noop(f"drink_potion_{color}", "missing_potion", {"color": color})
            return
        self.world.inventory["potions"][color] -= 1
        effect = self.world.potion_mapping[_potion_index(color)]
        if effect == 0:
            self.world.inventory["health"] = min(_max_stat(self.world, "health"), int(self.world.inventory["health"]) + 8)
        elif effect == 1:
            self.world.inventory["health"] = max(0, int(self.world.inventory["health"]) - 3)
        elif effect == 2:
            self.world.inventory["mana"] = min(_max_stat(self.world, "mana"), int(self.world.inventory["mana"]) + 8)
        elif effect == 3:
            self.world.inventory["mana"] = max(0, int(self.world.inventory["mana"]) - 3)
        elif effect == 4:
            self.world.inventory["energy"] = min(_max_stat(self.world, "energy"), int(self.world.inventory["energy"]) + 8)
        elif effect == 5:
            self.world.inventory["energy"] = max(0, int(self.world.inventory["energy"]) - 3)
        self._unlock("drink_potion")
        self._append_action(f"drink_potion_{color}", "drink_potion", {"color": color, "effect_index": effect, "potion_mapping": list(self.world.potion_mapping)})

    def _read_book(self) -> None:
        assert self.world is not None
        if int(self.world.inventory["books"]) <= 0:
            self._append_noop("read_book", "missing_book")
            return
        spells = self.world.inventory["learned_spells"]
        self.world.inventory["books"] -= 1
        learned_now: list[str] = []
        if not spells:
            spells.append("fireball")
            learned_now.append("fireball")
            self._unlock("learn_spell")
        self._append_action("read_book", "learn_spell", {"learned_spells": list(spells), "learned_now": learned_now})

    def _enchant(self, action: str) -> None:
        assert self.world is not None
        item = action.removeprefix("enchant_")
        target_tile = self.tile_at(self.front_pos()) if self.in_bounds(self.front_pos()) else "out_of_bounds"
        if target_tile not in {"enchantment_table_fire", "enchantment_table_ice"}:
            self._append_noop(action, "needs_enchantment_table", {"front_tile": target_tile})
            return
        if int(self.world.inventory["mana"]) < 9:
            self._append_noop(action, "needs_mana", {"mana": self.world.inventory["mana"]})
            return
        enchantment = "fire" if target_tile == "enchantment_table_fire" else "ice"
        gem = "ruby" if enchantment == "fire" else "sapphire"
        if int(self.world.inventory.get(gem, 0)) <= 0:
            self._append_noop(action, f"missing_{gem}", {"gem": gem})
            return
        if item == "sword" and int(self.world.inventory["sword"]) <= 0:
            self._append_noop(action, "missing_sword")
            return
        if item == "bow" and int(self.world.inventory["bow"]) <= 0:
            self._append_noop(action, "missing_bow")
            return
        if item == "armour" and sum(_armour(self.world)) <= 0:
            self._append_noop(action, "missing_armour", {"armour": _armour(self.world)})
            return
        armour_target_index: int | None = None
        if item == "armour":
            enchants = list(self.world.inventory.get("armour_enchantments", ["none", "none", "none", "none"]))
            unenchanted_targets = [idx for idx, value in enumerate(enchants) if value == "none"]
            opposite_targets = [idx for idx, value in enumerate(enchants) if value not in {"none", enchantment}]
            targets = unenchanted_targets or opposite_targets
            if not targets:
                self._append_noop(action, "no_valid_armour_target", {"armour_enchantments": enchants})
                return
            armour_target_index = self._rng.choice(targets)
        self.world.inventory[gem] -= 1
        self.world.inventory["mana"] -= 9
        if item == "armour":
            enchants = list(self.world.inventory.get("armour_enchantments", ["none", "none", "none", "none"]))
            enchants[int(armour_target_index)] = enchantment
            self.world.inventory["armour_enchantments"] = enchants
        else:
            self.world.inventory[f"{item}_enchantment"] = enchantment
        if action in {"enchant_sword", "enchant_armour"}:
            self._unlock(action)
        self._append_action(action, "enchant", {"item": item, "gem": gem, "enchantment": enchantment})

    def _level_up(self, action: str) -> None:
        assert self.world is not None
        attr = action.removeprefix("level_up_")
        if int(self.world.inventory["xp"]) <= 0:
            self._append_noop(action, "missing_xp", {"attribute": attr})
            return
        if int(self.world.inventory[attr]) >= 5:
            self._append_noop(action, "attribute_max", {"attribute": attr, "value": self.world.inventory[attr]})
            return
        self.world.inventory["xp"] -= 1
        self.world.inventory[attr] = int(self.world.inventory[attr]) + 1
        self._append_action(action, "level_up", {"attribute": attr, "value": self.world.inventory[attr]})

    def _melee(self, entity: Entity, action: str) -> None:
        assert self.world is not None
        # Entity transitions describe the combat result, not the submitted
        # action. Keep a primary action event so an action_applied-only replay
        # tape preserves this world-advancing step.
        self._append_action(action, "melee", {"target": list(entity.pos), "entity": entity.to_dict()})
        if entity.kind in PASSIVE_MOBS:
            damage = self._damage_against_entity(entity, self._player_damage_vector())
            entity.health -= damage
            if entity.health <= 0:
                entity.mask = False
                self.world.inventory["food"] = min(_max_stat(self.world, "food"), int(self.world.inventory["food"]) + 6)
                self.world.player_hunger = 0.0
                achievement = MOB_ACHIEVEMENTS.get(entity.kind)
                if achievement:
                    self._unlock(achievement)
                self._unlock("collect_food")
                self._append_entity(action, entity, "eat_passive")
            else:
                self._append_entity(action, entity, "damage")
            return
        self._damage_entity(entity, self._player_damage_vector(), action)

    def _damage_entity(self, entity: Entity, damage_vector: list[float], action: str) -> None:
        assert self.world is not None
        damage = self._damage_against_entity(entity, damage_vector)
        entity.health -= damage
        if entity.health <= 0:
            entity.mask = False
            if entity.kind in HOSTILE_MOBS:
                self.world.monsters_killed[self.world.player_level] = int(self.world.monsters_killed[self.world.player_level]) + 1
                achievement = MOB_ACHIEVEMENTS.get(entity.kind)
                if achievement:
                    self._unlock(achievement)
            self._append_entity(action, entity, "defeat")
        else:
            self._append_entity(action, entity, "damage")

    def _spawn_player_projectile(self, kind: str, action: str) -> bool:
        assert self.world is not None
        capacity = int(self.resolved.world.get("max_player_projectiles", 3)) if self.resolved else 3
        active_count = sum(1 for projectile in self.world.player_projectiles if projectile.mask and projectile.level == self.world.player_level)
        if active_count >= capacity:
            self._append_action(action, "noop", {"reason": "projectile_capacity_full", "capacity": capacity})
            return False
        slot_index = self._projectile_slot_index(self.world.player_projectiles)
        projectile = Projectile(
            id=f"player_projectile_{self.world.player_level}_{self.world.timestep}_{slot_index}",
            kind=kind,
            owner="player",
            level=self.world.player_level,
            pos=self.world.player_pos,
            direction=self.world.player_direction,
            mask=True,
        )
        for index, existing in enumerate(self.world.player_projectiles):
            if not existing.mask and existing.level == self.world.player_level:
                self.world.player_projectiles[index] = projectile
                break
        else:
            self.world.player_projectiles.append(projectile)
        self._append_projectile(action, projectile, "spawn", {"pos": list(projectile.pos), "direction": list(projectile.direction)})
        return True

    def _spawn_mob_projectile(self, entity: Entity, direction: tuple[int, int], action: str) -> bool:
        assert self.world is not None
        capacity = int(self.resolved.world.get("max_mob_projectiles", 3)) if self.resolved else 3
        active_count = sum(1 for projectile in self.world.mob_projectiles if projectile.mask and projectile.level == self.world.player_level)
        if active_count >= capacity:
            return False
        mob_type_id = MOB_TYPE_IDS.get(entity.kind, 0)
        projectile_type_id = RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_IDS[mob_type_id]
        slot_index = self._projectile_slot_index(self.world.mob_projectiles)
        projectile = Projectile(
            id=f"mob_projectile_{self.world.player_level}_{self.world.timestep}_{slot_index}",
            kind=PROJECTILE_TYPES[projectile_type_id],
            owner="mob",
            level=self.world.player_level,
            pos=entity.pos,
            direction=direction,
            mask=True,
        )
        for index, existing in enumerate(self.world.mob_projectiles):
            if not existing.mask and existing.level == self.world.player_level:
                self.world.mob_projectiles[index] = projectile
                break
        else:
            self.world.mob_projectiles.append(projectile)
        self._append_projectile(action, projectile, "spawn", {"source": entity.to_dict(), "pos": list(projectile.pos), "direction": list(projectile.direction)})
        return True

    def _projectile_slot_index(self, projectiles: list[Projectile]) -> int:
        assert self.world is not None
        for index, existing in enumerate(projectiles):
            if not existing.mask and existing.level == self.world.player_level:
                return index
        return len(projectiles)

    def _update_mobs(self, action: str) -> None:
        assert self.world is not None
        for entity in list(self.world.entities):
            if entity.mask and entity.level == self.world.player_level and (entity.mob_class or _mob_class(entity.kind)) == "melee":
                self._update_melee_mob(entity, action)
        for entity in list(self.world.entities):
            if entity.mask and entity.level == self.world.player_level and (entity.mob_class or _mob_class(entity.kind)) == "passive":
                self._update_passive_mob(entity, action)
        for entity in list(self.world.entities):
            if entity.mask and entity.level == self.world.player_level and (entity.mob_class or _mob_class(entity.kind)) == "ranged":
                self._update_ranged_mob(entity, action)

    def _update_melee_mob(self, entity: Entity, action: str) -> None:
        assert self.world is not None
        old_pos = entity.pos
        old_distance = _manhattan(old_pos, self.world.player_pos)
        random_direction = self._random_in_bounds_direction(old_pos, CARDINAL_DIRECTIONS)
        random_position = _add_pos(old_pos, random_direction)
        player_direction = self._mob_player_direction(old_pos)
        player_position = _add_pos(old_pos, player_direction)
        if old_distance < 10 and self._rng.random() < 0.75:
            proposed = player_position
        else:
            proposed = random_position

        attacking = old_distance == 1 and entity.attack_cooldown <= 0 and entity.mask
        if attacking:
            proposed = old_pos
            base_damage = list(MOB_TYPE_DAMAGE_MAPPING[MOB_TYPE_IDS.get(entity.kind, 0)][MOB_CLASSES.index("melee")])
            if self.world.is_sleeping:
                base_damage = [component * 3.5 for component in base_damage]
                self._unlock("wake_up")
            damage = self._damage_player(base_damage)
            self.world.is_sleeping = False
            self.world.is_resting = False
            entity.attack_cooldown = 5
            self._append_entity(action, entity, "attack_player")
            self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="combat", action=action, transition="mob_attack", message=f"MobAttack({entity.kind},{damage:.2f})", payload={"entity": entity.to_dict(), "damage": damage})
        else:
            entity.attack_cooldown -= 1

        if not attacking and self._valid_mob_position(entity, proposed, "melee"):
            entity.pos = proposed
        if old_distance >= self._mob_despawn_distance():
            entity.mask = False
            self._append_entity(action, entity, "despawn")

    def _update_passive_mob(self, entity: Entity, action: str) -> None:
        assert self.world is not None
        old_pos = entity.pos
        old_distance = _manhattan(old_pos, self.world.player_pos)
        proposed = _add_pos(old_pos, self._random_in_bounds_direction(old_pos, PASSIVE_DIRECTIONS))
        if self._valid_mob_position(entity, proposed, "passive"):
            entity.pos = proposed
        if old_distance >= self._mob_despawn_distance():
            entity.mask = False
            self._append_entity(action, entity, "despawn")

    def _update_ranged_mob(self, entity: Entity, action: str) -> None:
        assert self.world is not None
        old_pos = entity.pos
        old_distance = _manhattan(old_pos, self.world.player_pos)
        random_direction = self._random_in_bounds_direction(old_pos, CARDINAL_DIRECTIONS)
        random_position = _add_pos(old_pos, random_direction)
        player_direction = self._mob_player_direction(old_pos)
        selected_axis_distance = abs(self.world.player_pos[0] - old_pos[0]) if player_direction[0] != 0 else abs(self.world.player_pos[1] - old_pos[1])
        towards_player = _add_pos(old_pos, player_direction)
        away_from_player = (old_pos[0] - player_direction[0], old_pos[1] - player_direction[1])

        far_from_player = selected_axis_distance >= 6
        too_close_to_player = selected_axis_distance <= 3
        proposed = towards_player if far_from_player else random_position
        if too_close_to_player:
            proposed = away_from_player
        if self._rng.random() <= 0.85:
            proposed = random_position

        attacking = (not far_from_player) and entity.attack_cooldown <= 0 and entity.mask
        if attacking:
            self._spawn_mob_projectile(entity, player_direction, action)
            proposed = old_pos
            entity.attack_cooldown = 4
            self._append_entity(action, entity, "ranged_attack")
        else:
            entity.attack_cooldown -= 1

        if not attacking and self._valid_mob_position(entity, proposed, "ranged"):
            entity.pos = proposed
        if old_distance >= self._mob_despawn_distance():
            entity.mask = False
            self._append_entity(action, entity, "despawn")

    def _update_projectiles(self, action: str) -> None:
        self._update_mob_projectiles(action)
        self._update_player_projectiles(action)

    def _spawn_mobs(self, action: str) -> None:
        assert self.world is not None
        self._maybe_spawn_mob("passive", action)
        self._maybe_spawn_mob("melee", action)
        self._maybe_spawn_mob("ranged", action)

    def _maybe_spawn_mob(self, mob_class: str, action: str) -> None:
        assert self.world is not None
        class_index = MOB_CLASSES.index(mob_class)
        capacity = self._mob_capacity(mob_class)
        if self._active_mob_count(mob_class, self.world.player_level) >= capacity:
            return
        spawn_chance = FLOOR_MOB_SPAWN_CHANCE[self.world.player_level][class_index]
        if mob_class == "melee":
            spawn_chance += FLOOR_MOB_SPAWN_CHANCE[self.world.player_level][3] * (1.0 - self.world.light_level) ** 2
        if mob_class in {"melee", "ranged"}:
            spawn_chance *= self._monster_spawn_coeff()
        if self._is_fighting_boss() and mob_class == "passive":
            return
        if self._rng.random() >= spawn_chance:
            return
        kind = self._spawn_mob_kind(mob_class)
        candidates = self._spawn_candidates(mob_class, kind)
        if not candidates:
            return
        pos = self._rng.choice(candidates)
        entity = self._reuse_or_create_entity(kind, mob_class, pos)
        self._append_entity(action, entity, "spawn")

    def _spawn_candidates(self, mob_class: str, kind: str) -> list[tuple[int, int]]:
        assert self.world is not None
        candidates: list[tuple[int, int]] = []
        for y, row in enumerate(self.world.maps[self.world.player_level]):
            for x, raw_block in enumerate(row):
                pos = (x, y)
                block = normalize_tile(raw_block)
                distance = _euclidean(pos, self.world.player_pos)
                if self.entity_at(pos, self.world.player_level) is not None:
                    continue
                if self.projectile_at(pos, self.world.player_level) is not None:
                    continue
                if pos == self.world.player_pos:
                    continue
                if mob_class == "passive":
                    if block not in {"grass", "path", "fire_grass", "ice_grass"}:
                        continue
                    if distance <= 3 or distance >= self._mob_despawn_distance():
                        continue
                else:
                    if self._is_fighting_boss():
                        if block not in {"grave", "grave2", "grave3"}:
                            continue
                        if distance > 6:
                            continue
                    else:
                        if kind == "deep_thing":
                            if block != "water":
                                continue
                        elif block not in {"grass", "path", "fire_grass", "ice_grass"}:
                            continue
                        if distance <= 9 or distance >= self._mob_despawn_distance():
                            continue
                candidates.append(pos)
        return candidates

    def _reuse_or_create_entity(self, kind: str, mob_class: str, pos: tuple[int, int]) -> Entity:
        assert self.world is not None
        for entity in self.world.entities:
            if not entity.mask and entity.level == self.world.player_level and (entity.mob_class or _mob_class(entity.kind)) == mob_class:
                entity.id = f"{kind}_{self.world.timestep}_{len(self.world.entities)}"
                entity.kind = kind
                entity.mob_class = mob_class
                entity.pos = pos
                entity.health = float(MOB_HEALTH[kind])
                entity.attack_cooldown = 0
                entity.mask = True
                return entity
        entity = Entity(
            id=f"{kind}_{self.world.timestep}_{len(self.world.entities)}",
            kind=kind,
            pos=pos,
            health=float(MOB_HEALTH[kind]),
            level=self.world.player_level,
            mob_class=mob_class,
        )
        self.world.entities.append(entity)
        return entity

    def _spawn_mob_kind(self, mob_class: str) -> str:
        assert self.world is not None
        level = self.world.player_level
        if self._is_fighting_boss():
            level = max(0, min(self.world.levels - 1, int(self.world.inventory.get("boss_progress", 0))))
        choices = FLOOR_MOBS.get(level, FLOOR_MOBS[0])
        return choices[{"passive": 0, "melee": 1, "ranged": 2}[mob_class]]

    def _monster_spawn_coeff(self) -> float:
        assert self.world is not None
        coeff = 1.0 + (2.0 if self.world.monsters_killed[self.world.player_level] < MONSTERS_KILLED_TO_CLEAR_LEVEL else 0.0)
        if self._is_fighting_boss():
            coeff *= 1000.0 if self.world.boss_timesteps_to_spawn_this_round >= 1 else 0.0
        return coeff

    def _mob_capacity(self, mob_class: str) -> int:
        assert self.resolved is not None
        defaults = {"passive": 3, "melee": 3, "ranged": 2}
        return int(self.resolved.world.get(f"max_{mob_class}_mobs", defaults[mob_class]))

    def _active_mob_count(self, mob_class: str, level: int) -> int:
        assert self.world is not None
        return sum(1 for entity in self.world.entities if entity.mask and entity.level == level and (entity.mob_class or _mob_class(entity.kind)) == mob_class)

    def _update_player_projectiles(self, action: str) -> None:
        assert self.world is not None
        for projectile in self.world.player_projectiles:
            if not projectile.mask or projectile.level != self.world.player_level:
                continue
            old_pos = projectile.pos
            proposed = (old_pos[0] + projectile.direction[0], old_pos[1] + projectile.direction[1])
            damage_vector = self._projectile_damage_vector(projectile.kind)
            current_entity = self.entity_at(old_pos, projectile.level)
            if current_entity is not None:
                self._damage_entity(current_entity, damage_vector, action)
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "hit_mob", {"from": list(old_pos), "hit_pos": list(old_pos), "to": list(proposed), "entity": current_entity.to_dict()})
                continue
            proposed_entity = self.entity_at(proposed, projectile.level) if self.in_bounds(proposed) else None
            if proposed_entity is not None:
                self._damage_entity(proposed_entity, damage_vector, action)
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "hit_mob", {"from": list(old_pos), "hit_pos": list(proposed), "to": list(proposed), "entity": proposed_entity.to_dict()})
                continue
            if not self.in_bounds(proposed):
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "despawn", {"from": list(old_pos), "to": list(proposed), "reason": "out_of_bounds"})
                continue
            if self.block_at(proposed) in STATIC_SOLID_BLOCKS:
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "despawn", {"from": list(old_pos), "to": list(proposed), "reason": f"blocked:{self.block_at(proposed)}"})
                continue
            projectile.pos = proposed
            self._append_projectile(action, projectile, "move", {"from": list(old_pos), "to": list(proposed)})

    def _update_mob_projectiles(self, action: str) -> None:
        assert self.world is not None
        for projectile in self.world.mob_projectiles:
            if not projectile.mask or projectile.level != self.world.player_level:
                continue
            old_pos = projectile.pos
            proposed = (old_pos[0] + projectile.direction[0], old_pos[1] + projectile.direction[1])
            hit_player = old_pos == self.world.player_pos or proposed == self.world.player_pos
            if hit_player:
                damage = self._damage_player(self._mob_projectile_damage_vector(projectile.kind))
                self.world.is_sleeping = False
                self.world.is_resting = False
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "hit_player", {"from": list(old_pos), "to": list(proposed), "damage": damage})
                continue
            if not self.in_bounds(proposed):
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "despawn", {"from": list(old_pos), "to": list(proposed), "reason": "out_of_bounds"})
                continue
            if self.block_at(proposed) in STATIC_SOLID_BLOCKS:
                hit_breakable = self.block_at(proposed) in {"furnace", "crafting_table"}
                if hit_breakable:
                    self._set_tile(proposed, "path")
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "despawn", {"from": list(old_pos), "to": list(proposed), "reason": f"blocked:{self.block_at(proposed)}", "removed_block": hit_breakable})
                continue
            if self.entity_at(proposed, projectile.level) is not None:
                projectile.pos = proposed
                projectile.mask = False
                self._append_projectile(action, projectile, "despawn", {"from": list(old_pos), "to": list(proposed), "reason": "blocked_by_mob"})
                continue
            projectile.pos = proposed
            self._append_projectile(action, projectile, "move", {"from": list(old_pos), "to": list(proposed)})

    def _damage_player(self, damage_vector: list[float]) -> float:
        assert self.world is not None
        armour = _armour(self.world)
        armour_enchantments = list(self.world.inventory.get("armour_enchantments", ["none", "none", "none", "none"]))
        defense_vector = [
            sum(piece * 0.1 for piece in armour),
            sum(0.2 for enchantment in armour_enchantments if enchantment == "fire"),
            sum(0.2 for enchantment in armour_enchantments if enchantment == "ice"),
        ]
        damage = sum((1.0 - defense_vector[index]) * damage_vector[index] for index in range(3))
        self.world.inventory["health"] = float(self.world.inventory["health"]) - damage
        return damage

    def _player_damage_vector(self) -> list[float]:
        assert self.world is not None
        inv = self.world.inventory
        physical_damages = [1.0, 2.0, 3.0, 5.0, 8.0]
        sword_tier = max(0, min(4, int(inv.get("sword", 0))))
        physical_damage = physical_damages[sword_tier]
        fire_damage = 0.0
        ice_damage = 0.0
        enchantment = str(inv.get("sword_enchantment", "none"))
        if enchantment == "fire":
            fire_damage = physical_damage * 0.5
        elif enchantment == "ice":
            ice_damage = physical_damage * 0.5
        physical_damage *= 1.0 + 0.25 * (int(inv.get("strength", 1)) - 1)
        enchantment_coeff = 1.0 + 0.05 * (int(inv.get("intelligence", 1)) - 1)
        fire_damage *= enchantment_coeff
        ice_damage *= enchantment_coeff
        return [physical_damage, fire_damage, ice_damage]

    def _projectile_damage_vector(self, projectile: str) -> list[float]:
        assert self.world is not None
        projectile_type_id = PROJECTILE_DAMAGE_TYPE_IDS[projectile]
        vector = list(MOB_TYPE_DAMAGE_MAPPING[projectile_type_id][MOB_CLASSES.index("projectile")])
        if projectile in {"arrow", "arrow2"}:
            enchantment = str(self.world.inventory.get("bow_enchantment", "none"))
            if enchantment == "fire":
                vector[1] += vector[0] / 2.0
            elif enchantment == "ice":
                vector[2] += vector[0] / 2.0
            vector = [component * (1.0 + 0.2 * (int(self.world.inventory.get("dexterity", 1)) - 1)) for component in vector]
        elif projectile in {"fireball", "iceball"}:
            vector = [component * (1.0 + 0.5 * (int(self.world.inventory.get("intelligence", 1)) - 1)) for component in vector]
        return vector

    def _mob_projectile_damage_vector(self, projectile: str) -> list[float]:
        projectile_type_id = PROJECTILE_DAMAGE_TYPE_IDS[projectile]
        return list(MOB_TYPE_DAMAGE_MAPPING[projectile_type_id][MOB_CLASSES.index("projectile")])

    def _damage_against_entity(self, entity: Entity, damage_vector: list[float]) -> float:
        mob_type_id = MOB_TYPE_IDS.get(entity.kind, 0)
        class_id = MOB_CLASSES.index(entity.mob_class or _mob_class(entity.kind))
        defense = MOB_TYPE_DEFENSE_MAPPING[mob_type_id][class_id]
        return sum((1.0 - defense[index]) * damage_vector[index] for index in range(3))

    def _update_intrinsics(self, action: str) -> None:
        assert self.world is not None
        inv = self.world.inventory

        if action == "sleep" and float(inv["energy"]) < _max_stat(self.world, "energy"):
            self.world.is_sleeping = True

        if float(inv["energy"]) >= _max_stat(self.world, "energy") and self.world.is_sleeping:
            self.world.is_sleeping = False
            self._unlock("wake_up")

        if action == "rest" and float(inv["health"]) < _max_stat(self.world, "health"):
            self.world.is_resting = True

        if self.world.is_resting and (
            float(inv["health"]) >= _max_stat(self.world, "health")
            or float(inv["food"]) <= 0
            or float(inv["drink"]) <= 0
        ):
            self.world.is_resting = False

        not_boss = self.world.player_level != self.world.levels - 1
        decay_coeff = 1.0 - (0.125 * (int(inv.get("dexterity", 1)) - 1))

        hunger_add = (0.5 if self.world.is_sleeping else 1.0) * decay_coeff
        self.world.player_hunger += hunger_add
        if self.world.player_hunger > 25:
            if not_boss:
                inv["food"] = max(float(inv["food"]) - 1, 0)
            self.world.player_hunger = 0.0

        thirst_add = (0.5 if self.world.is_sleeping else 1.0) * decay_coeff
        self.world.player_thirst += thirst_add
        if self.world.player_thirst > 20:
            if not_boss:
                inv["drink"] = max(float(inv["drink"]) - 1, 0)
            self.world.player_thirst = 0.0

        if self.world.is_sleeping:
            self.world.player_fatigue = min(self.world.player_fatigue - 1.0, 0.0)
        else:
            self.world.player_fatigue += decay_coeff
        if self.world.player_fatigue > 30:
            if not_boss:
                inv["energy"] = max(float(inv["energy"]) - 1, 0)
            self.world.player_fatigue = 0.0
        if self.world.player_fatigue < -10:
            inv["energy"] = min(float(inv["energy"]) + 1, _max_stat(self.world, "energy"))
            self.world.player_fatigue = 0.0

        all_necessities = float(inv["food"]) > 0 and float(inv["drink"]) > 0 and (float(inv["energy"]) > 0 or self.world.is_sleeping)
        if all_necessities:
            recover_add = (2.0 if self.world.is_sleeping else 1.0) * 2.0
        else:
            recover_add = ( -0.5 if self.world.is_sleeping else -1.0) if not_boss else 0.0
        self.world.player_recover += recover_add
        if self.world.player_recover > 25:
            inv["health"] = min(float(inv["health"]) + 1, _max_stat(self.world, "health"))
            self.world.player_recover = 0.0
        if self.world.player_recover < -15:
            inv["health"] = float(inv["health"]) - 1
            self.world.player_recover = 0.0

        mana_recover_coeff = 1.0 + 0.25 * (int(inv.get("intelligence", 1)) - 1)
        self.world.player_recover_mana = (self.world.player_recover_mana + (2.0 if self.world.is_sleeping else 1.0)) * mana_recover_coeff
        if self.world.player_recover_mana > 30:
            inv["mana"] = float(inv["mana"]) + 1
            self.world.player_recover_mana = 0.0

    def _clip_inventory_and_intrinsics(self) -> None:
        assert self.world is not None
        inv = self.world.inventory
        for key, value in list(inv.items()):
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                inv[key] = min(value, 99)
            elif isinstance(value, float):
                inv[key] = min(value, 99.0)
            elif isinstance(value, dict):
                inv[key] = {sub_key: min(int(sub_value), 99) for sub_key, sub_value in value.items()}
            elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
                inv[key] = [min(int(item), 99) for item in value]
        min_health = 9 if bool(self.resolved and self.resolved.rules.get("god_mode", False)) else 0
        inv["health"] = min(max(float(inv["health"]), min_health), _max_stat(self.world, "health"))
        inv["food"] = min(max(float(inv["food"]), 0), _max_stat(self.world, "food"))
        inv["drink"] = min(max(float(inv["drink"]), 0), _max_stat(self.world, "drink"))
        inv["energy"] = min(max(float(inv["energy"]), 0), _max_stat(self.world, "energy"))
        inv["mana"] = min(max(float(inv["mana"]), 0), _max_stat(self.world, "mana"))

    def _calculate_inventory_achievements(self) -> None:
        assert self.world is not None
        inv = self.world.inventory
        resource_achievements = {
            "wood": "collect_wood",
            "stone": "collect_stone",
            "coal": "collect_coal",
            "iron": "collect_iron",
            "diamond": "collect_diamond",
            "ruby": "collect_ruby",
            "sapphire": "collect_sapphire",
            "sapling": "collect_sapling",
        }
        for resource, achievement in resource_achievements.items():
            if int(inv.get(resource, 0)) > 0:
                self._unlock(achievement)
        if int(inv.get("bow", 0)) > 0:
            self._unlock("find_bow")
        if int(inv.get("arrows", 0)) > 0:
            self._unlock("make_arrow")
        if int(inv.get("torches", 0)) > 0:
            self._unlock("make_torch")

        pickaxe = int(inv.get("pickaxe", 0))
        sword = int(inv.get("sword", 0))
        for threshold, achievement in (
            (1, "make_wood_pickaxe"),
            (2, "make_stone_pickaxe"),
            (3, "make_iron_pickaxe"),
            (4, "make_diamond_pickaxe"),
        ):
            if pickaxe >= threshold:
                self._unlock(achievement)
        for threshold, achievement in (
            (1, "make_wood_sword"),
            (2, "make_stone_sword"),
            (3, "make_iron_sword"),
            (4, "make_diamond_sword"),
        ):
            if sword >= threshold:
                self._unlock(achievement)

    def _done_reason(self) -> str | None:
        assert self.world is not None and self.resolved is not None
        if float(self.world.inventory["health"]) <= 0:
            return "death"
        if self.world.achievements.get("defeat_necromancer", 0) > 0:
            return "boss_defeated"
        if self.world.timestep >= self.resolved.max_steps:
            return "max_steps"
        return None

    def _public_state(self, *, done: bool) -> PublicState:
        assert self.world is not None
        return PublicState(
            observation=self._observation(),
            player_pos=self.world.player_pos,
            level=self.world.player_level,
            inventory=copy.deepcopy(self.world.inventory),
            achievements=dict(self.world.achievements),
            done=done,
        )

    def _observation(self) -> dict[str, Any]:
        assert self.world is not None
        return {
            "player": {
                "pos": [self.world.player_pos[0], self.world.player_pos[1]],
                "level": self.world.player_level,
                "direction": [self.world.player_direction[0], self.world.player_direction[1]],
                "front_tile": self.tile_at(self.front_pos()) if self.in_bounds(self.front_pos()) else "out_of_bounds",
                "front_block": self.block_at(self.front_pos()) if self.in_bounds(self.front_pos()) else "out_of_bounds",
                "front_item": self.item_at(self.front_pos()) if self.in_bounds(self.front_pos()) else "none",
            },
            "local_map": self.local_map(),
            "inventory": copy.deepcopy(self.world.inventory),
            "intrinsics": {
                "is_sleeping": self.world.is_sleeping,
                "is_resting": self.world.is_resting,
                "recover": self.world.player_recover,
                "hunger": self.world.player_hunger,
                "thirst": self.world.player_thirst,
                "fatigue": self.world.player_fatigue,
                "recover_mana": self.world.player_recover_mana,
            },
            "potion_mapping": list(self.world.potion_mapping),
            "floor_state": {
                "monsters_killed": list(self.world.monsters_killed),
                "chests_opened": list(self.world.chests_opened),
                "down_ladders": [[x, y] for x, y in self.world.down_ladders],
                "up_ladders": [[x, y] for x, y in self.world.up_ladders],
                "growing_plants": copy.deepcopy(self.world.growing_plants),
            },
            "mob_state": self._mob_state_readout(),
            "projectile_state": self._projectile_state_readout(),
            "achievements": sorted(name for name, count in self.world.achievements.items() if count > 0),
            "nearby_entities": [entity.to_dict() for entity in self.entities_near(radius=5)],
        }

    def local_map(self, radius: int | None = None) -> list[str]:
        assert self.world is not None and self.resolved is not None
        r = int(radius if radius is not None else self.resolved.world.get("view_radius", 4))
        rows: list[str] = []
        px, py = self.world.player_pos
        for y in range(py - r, py + r + 1):
            chars = []
            for x in range(px - r, px + r + 1):
                if (x, y) == self.world.player_pos:
                    chars.append("P")
                    continue
                entity = self.entity_at((x, y), self.world.player_level)
                if entity is not None:
                    chars.append(_entity_char(entity.kind))
                    continue
                projectile = self.projectile_at((x, y), self.world.player_level)
                if projectile is not None:
                    chars.append(_projectile_char(projectile))
                    continue
                tile = self.tile_at((x, y)) if self.in_bounds((x, y)) else "out_of_bounds"
                chars.append(_tile_char(tile))
            rows.append("".join(chars))
        return rows

    def _ascii_map(self) -> str:
        assert self.world is not None
        rows: list[str] = []
        for y in range(self.world.height):
            chars = []
            for x in range(self.world.width):
                if (x, y) == self.world.player_pos:
                    chars.append("P")
                    continue
                entity = self.entity_at((x, y), self.world.player_level)
                if entity is not None:
                    chars.append(_entity_char(entity.kind))
                    continue
                projectile = self.projectile_at((x, y), self.world.player_level)
                if projectile is not None:
                    chars.append(_projectile_char(projectile))
                    continue
                chars.append(_tile_char(self.tile_at((x, y))))
            rows.append("".join(chars))
        return "\n".join(rows)

    def in_bounds(self, pos: tuple[int, int]) -> bool:
        assert self.world is not None
        return 0 <= pos[0] < self.world.width and 0 <= pos[1] < self.world.height

    def tile_at(self, pos: tuple[int, int]) -> str:
        assert self.world is not None
        if not self.in_bounds(pos):
            return "out_of_bounds"
        item = self.item_at(pos)
        if item != "none":
            return item
        return self.block_at(pos)

    def block_at(self, pos: tuple[int, int]) -> str:
        assert self.world is not None
        if not self.in_bounds(pos):
            return "out_of_bounds"
        return normalize_tile(self.world.maps[self.world.player_level][pos[1]][pos[0]])

    def item_at(self, pos: tuple[int, int]) -> str:
        assert self.world is not None
        if not self.in_bounds(pos):
            return "none"
        return normalize_item(self.world.item_maps[self.world.player_level][pos[1]][pos[0]])

    def _set_tile(self, pos: tuple[int, int], kind: str) -> None:
        assert self.world is not None
        self._set_cell(self.world, self.world.player_level, pos, kind)

    def _set_item(self, pos: tuple[int, int], kind: str) -> None:
        assert self.world is not None
        self._set_item_cell(self.world, self.world.player_level, pos, kind)

    def _set_cell(self, world: CraftaxWorld, level: int, pos: tuple[int, int], kind: str) -> None:
        normalized = normalize_tile(kind)
        if normalized in ITEM_OVERLAYS or normalized == "none":
            self._set_item_cell(world, level, pos, normalized)
            return
        world.maps[level][pos[1]][pos[0]] = normalized
        if normalized in SOLID:
            world.item_maps[level][pos[1]][pos[0]] = "none"

    def _set_item_cell(self, world: CraftaxWorld, level: int, pos: tuple[int, int], kind: str) -> None:
        item = normalize_item(kind)
        if item in {"ladder_down", "ladder_up"}:
            for row in world.item_maps[level]:
                for idx, existing in enumerate(row):
                    if existing == item:
                        row[idx] = "none"
        world.item_maps[level][pos[1]][pos[0]] = item
        if item == "ladder_down":
            world.down_ladders[level] = pos
        elif item == "ladder_up":
            world.up_ladders[level] = pos

    def _add_torch_light(self, pos: tuple[int, int]) -> None:
        assert self.world is not None
        level = self.world.player_level
        cx, cy = pos
        for y in range(max(0, cy - 4), min(self.world.height, cy + 5)):
            for x in range(max(0, cx - 4), min(self.world.width, cx + 5)):
                distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                contribution = max(0.0, min(1.0, 1.0 - distance / 5.0))
                self.world.light_maps[level][y][x] = max(0.0, min(1.0, self.world.light_maps[level][y][x] + contribution))

    def _add_growing_plant(self, pos: tuple[int, int]) -> None:
        assert self.world is not None
        entry = {"pos": [pos[0], pos[1]], "age": 0, "active": True}
        for index, existing in enumerate(self.world.growing_plants):
            if not _strict_bool(existing.get("active", False), f"growing_plants[{index}].active"):
                self.world.growing_plants[index] = entry
                return
        if len(self.world.growing_plants) >= MAX_GROWING_PLANTS:
            return
        self.world.growing_plants.append(entry)

    def _reset_growing_plant(self, pos: tuple[int, int]) -> None:
        assert self.world is not None
        for index, plant in enumerate(self.world.growing_plants):
            if _pos(plant.get("pos", (-1, -1))) == pos and _strict_bool(plant.get("active", False), f"growing_plants[{index}].active"):
                plant["age"] = 0
                return

    def _update_plants(self) -> None:
        assert self.world is not None
        for index, plant in enumerate(self.world.growing_plants):
            if not _strict_bool(plant.get("active", False), f"growing_plants[{index}].active"):
                continue
            pos = _pos(plant.get("pos", (-1, -1)))
            if not (0 <= pos[0] < self.world.width and 0 <= pos[1] < self.world.height):
                continue
            plant["age"] = int(plant.get("age", 0)) + 1
            if plant["age"] >= 600 and self.world.maps[0][pos[1]][pos[0]] == "plant":
                self.world.maps[0][pos[1]][pos[0]] = "ripe_plant"

    def front_pos(self) -> tuple[int, int]:
        assert self.world is not None
        return (self.world.player_pos[0] + self.world.player_direction[0], self.world.player_pos[1] + self.world.player_direction[1])

    def entity_at(self, pos: tuple[int, int], level: int) -> Entity | None:
        assert self.world is not None
        for entity in self.world.entities:
            if entity.mask and entity.level == level and entity.pos == pos:
                return entity
        return None

    def projectile_at(self, pos: tuple[int, int], level: int) -> Projectile | None:
        assert self.world is not None
        for projectile in [*self.world.player_projectiles, *self.world.mob_projectiles]:
            if projectile.mask and projectile.level == level and projectile.pos == pos:
                return projectile
        return None

    def _valid_mob_position(self, entity: Entity, pos: tuple[int, int], mob_class: str) -> bool:
        assert self.world is not None
        if not self.in_bounds(pos):
            return False
        if pos == self.world.player_pos:
            return False
        if self.entity_at(pos, entity.level) is not None:
            return False
        if self.projectile_at(pos, entity.level) is not None:
            return False
        block = self.block_at(pos)
        if block in STATIC_SOLID_BLOCKS:
            return False
        mob_type_id = MOB_TYPE_IDS.get(entity.kind, 0)
        class_id = MOB_CLASSES.index(mob_class)
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type_id][class_id]
        in_water = block == "water"
        in_lava = block == "lava"
        on_ground = block not in STATIC_SOLID_BLOCKS and not in_water and not in_lava
        if collision_map[0] and on_ground:
            return False
        if collision_map[1] and in_water:
            return False
        if collision_map[2] and in_lava:
            return False
        return True

    def _mob_player_direction(self, pos: tuple[int, int]) -> tuple[int, int]:
        assert self.world is not None
        dx = self.world.player_pos[0] - pos[0]
        dy = self.world.player_pos[1] - pos[1]
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        if abs_dx == 0 and abs_dy == 0:
            return (0, 0)
        if abs_dx == abs_dy:
            axis = self._rng.choice([0, 1])
        else:
            axis = 0 if abs_dx > abs_dy else 1
        if axis == 0:
            return (_sign(dx), 0)
        return (0, _sign(dy))

    def _random_in_bounds_direction(self, pos: tuple[int, int], directions: list[tuple[int, int]]) -> tuple[int, int]:
        candidates = [direction for direction in directions if self.in_bounds(_add_pos(pos, direction))]
        if not candidates:
            return (0, 0)
        return self._rng.choice(candidates)

    def _day_length(self) -> int:
        assert self.resolved is not None
        return _day_length_from_resolved(self.resolved)

    def _mob_despawn_distance(self) -> int:
        return int(self.resolved.world.get("mob_despawn_distance", 14)) if self.resolved else 14

    def _is_fighting_boss(self) -> bool:
        assert self.world is not None
        return self.world.player_level == self.world.levels - 1

    def _is_boss_vulnerable(self) -> bool:
        assert self.world is not None
        return (
            self._active_mob_count("melee", self.world.player_level) == 0
            and self._active_mob_count("ranged", self.world.player_level) == 0
            and self.world.boss_timesteps_to_spawn_this_round <= 0
        )

    def _update_boss_logic(self) -> None:
        assert self.world is not None
        if self._is_fighting_boss():
            self.world.boss_timesteps_to_spawn_this_round -= 1
        if int(self.world.inventory.get("boss_progress", 0)) >= self.world.levels - 1:
            self._unlock("defeat_necromancer")

    def entity_in_ray(self, max_range: int) -> Entity | None:
        assert self.world is not None
        dx, dy = self.world.player_direction
        x, y = self.world.player_pos
        for distance in range(1, max_range + 1):
            target = (x + dx * distance, y + dy * distance)
            entity = self.entity_at(target, self.world.player_level)
            if entity is not None:
                return entity
            if not self.in_bounds(target) or self.block_at(target) in STATIC_SOLID_BLOCKS:
                return None
        return None

    def entities_near(self, radius: int) -> list[Entity]:
        assert self.world is not None
        out: list[Entity] = []
        px, py = self.world.player_pos
        for entity in self.world.entities:
            if not entity.mask or entity.level != self.world.player_level:
                continue
            ex, ey = entity.pos
            if abs(ex - px) <= radius and abs(ey - py) <= radius:
                out.append(copy.deepcopy(entity))
        return out

    def _mob_state_readout(self) -> dict[str, list[dict[str, Any]]]:
        assert self.world is not None
        grouped: dict[str, list[dict[str, Any]]] = {"passive": [], "melee": [], "ranged": []}
        for entity in self.world.entities:
            if not entity.mask:
                continue
            grouped.setdefault(entity.mob_class or _mob_class(entity.kind), []).append(entity.to_dict())
        return grouped

    def _projectile_state_readout(self) -> dict[str, list[dict[str, Any]]]:
        assert self.world is not None
        return {
            "player": [projectile.to_dict() for projectile in self.world.player_projectiles if projectile.mask],
            "mob": [projectile.to_dict() for projectile in self.world.mob_projectiles if projectile.mask],
        }

    def near_tile(self, kinds: set[str]) -> bool:
        assert self.world is not None
        px, py = self.world.player_pos
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                pos = (px + dx, py + dy)
                if self.in_bounds(pos) and self.block_at(pos) in kinds:
                    return True
        return False

    def _pay(self, costs: dict[str, int]) -> bool:
        assert self.world is not None
        if any(int(self.world.inventory.get(key, 0)) < amount for key, amount in costs.items()):
            return False
        for key, amount in costs.items():
            self.world.inventory[key] -= amount
        return True

    def _unlock(self, name: str) -> None:
        assert self.world is not None
        if self.world.achievements.get(name, 0) <= 0:
            self.world.achievements[name] = 1
            delta = self._achievement_reward(name)
            self.private.reward_last += delta
            self.private.total_reward += delta

    def _achievement_reward(self, name: str) -> float:
        del name
        return 1.0

    def _apply_step_reward(self) -> None:
        reward = float(self.resolved.rules.get("step_reward", 0.0)) if self.resolved else 0.0
        if reward == 0.0:
            return
        self.private.reward_last += reward
        self.private.total_reward += reward

    def _apply_health_reward(self, before_health: float) -> None:
        assert self.world is not None
        if before_health <= 0.0:
            return
        health_delta = float(self.world.inventory.get("health", 0.0)) - before_health
        if health_delta == 0.0:
            return
        reward = health_delta * 0.1
        self.private.reward_last += reward
        self.private.total_reward += reward

    def _newly_unlocked(self, before: dict[str, int]) -> list[str]:
        assert self.world is not None
        return sorted(name for name, count in self.world.achievements.items() if count > 0 and before.get(name, 0) <= 0)

    def _append_achievement(self, action: str, achievement: str) -> None:
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="achievement_unlocked", action=action, transition=achievement, message=f"AchievementUnlocked({achievement})", payload={"achievement": achievement})

    def _append_action(self, action: str, transition: str, payload: dict[str, Any]) -> None:
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="action_applied", action=action, transition=transition, message=f"ActionApplied({action},step={self.private.step_index})", payload=payload)

    def _append_noop(self, action: str, reason: str, payload: dict[str, Any] | None = None) -> None:
        event_payload = {"reason": reason}
        if payload:
            event_payload.update(payload)
        self._append_action(action, "noop", event_payload)

    def _append_entity(self, action: str, entity: Entity, transition: str) -> None:
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="entity_transition", action=action, transition=transition, message=f"EntityTransition({transition},{entity.kind})", payload={"entity": entity.to_dict()})

    def _append_projectile(self, action: str, projectile: Projectile, transition: str, payload: dict[str, Any]) -> None:
        event_payload = {"projectile": projectile.to_dict()}
        event_payload.update(payload)
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="projectile_transition", action=action, transition=transition, message=f"ProjectileTransition({transition},{projectile.kind})", payload=event_payload)

    def _append_inventory_deltas(self, action: str, before: dict[str, Any], after: dict[str, Any]) -> None:
        for key in sorted(set(before) | set(after)):
            if key in {"potions", "learned_spells"}:
                if before.get(key) != after.get(key):
                    self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="resource_delta", action=action, transition=key, message=f"ResourceDelta({key})", payload={"before": before.get(key), "after": after.get(key)})
                continue
            if before.get(key) != after.get(key):
                delta = None
                if isinstance(before.get(key), (int, float)) and isinstance(after.get(key), (int, float)):
                    delta = after.get(key, 0) - before.get(key, 0)
                self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="resource_delta", action=action, transition=key, message=f"ResourceDelta({key},{delta})", payload={"resource": key, "before": before.get(key), "after": after.get(key), "delta": delta})

    def _reject(self, action: str, code: str) -> None:
        self.private.invalid_action_count += 1
        penalty = float(self.resolved.rules.get("invalid_action_penalty", -0.05)) if self.resolved else -0.05
        self.private.reward_last += penalty
        self.private.total_reward += penalty
        self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="rule_violation", action=action, transition="reject", severity="warn", message=f"RuleViolation({code})", payload={"code": code, "action": action})

    def _reject_terminal(self, action: str) -> None:
        self.nev.append(
            step_index=self.private.step_index,
            episode_id=self.private.episode_id,
            kind="rule_violation",
            action=action,
            transition="reject",
            severity="warn",
            message="RuleViolation(terminal)",
            payload={"code": "terminal", "action": action},
        )

    def _wake_from_intrinsic(self, action: str) -> None:
        was_sleeping = self.world.is_sleeping
        was_resting = self.world.is_resting
        self.world.is_sleeping = False
        self.world.is_resting = False
        if was_sleeping:
            self._unlock("wake_up")
        self.nev.append(
            step_index=self.private.step_index,
            episode_id=self.private.episode_id,
            kind="action_applied",
            action=action,
            transition="wake_from_intrinsic",
            message=f"ActionApplied({action},wake_from_intrinsic,step={self.private.step_index})",
            payload={"was_sleeping": was_sleeping, "was_resting": was_resting},
        )

    def _checkpoint_cadence_event(self) -> None:
        if self.resolved is None:
            return
        interval = int(self.resolved.world.get("checkpoint_every_n_steps", 0) or 0)
        if interval > 0 and self.private.step_index > 0 and self.private.step_index % interval == 0:
            self.nev.append(step_index=self.private.step_index, episode_id=self.private.episode_id, kind="checkpoint_cadence", transition="cadence", message=f"CheckpointCadence(step={self.private.step_index})", payload={"step_index": self.private.step_index, "nev_cursor": self.nev.cursor()})

    def _assert_runtime_invariants(self, context: str) -> None:
        if self.world is None or self.resolved is None:
            return
        world = self.world
        errors: list[str] = []

        def fail(message: str) -> None:
            errors.append(message)

        if world.width != self.resolved.width:
            fail(f"world.width={world.width} resolved.width={self.resolved.width}")
        if world.height != self.resolved.height:
            fail(f"world.height={world.height} resolved.height={self.resolved.height}")
        if world.max_steps != self.resolved.max_steps:
            fail(f"world.max_steps={world.max_steps} resolved.max_steps={self.resolved.max_steps}")
        if world.seed != self.resolved.seed:
            fail(f"world.seed={world.seed} resolved.seed={self.resolved.seed}")
        if world.timestep != self.private.step_index:
            fail(f"world.timestep={world.timestep} private.step_index={self.private.step_index}")
        if self.private.seed != self.resolved.seed:
            fail(f"private.seed={self.private.seed} resolved.seed={self.resolved.seed}")
        if self.private.task_id and self.private.task_id != self.resolved.task_id:
            fail(f"private.task_id={self.private.task_id!r} resolved.task_id={self.resolved.task_id!r}")
        if self.private.scenario_id and self.private.scenario_id != self.resolved.scenario_id:
            fail(f"private.scenario_id={self.private.scenario_id!r} resolved.scenario_id={self.resolved.scenario_id!r}")
        if self.private.episode_id and self.private.episode_id != self.resolved.episode_id:
            fail("private.episode_id does not match resolved episode_id")
        if self.private.config_hash and self.private.config_hash != self.resolved.config_hash:
            fail("private.config_hash does not match resolved config_hash")
        if not isinstance(self.private.step_index, int) or isinstance(self.private.step_index, bool) or self.private.step_index < 0:
            fail(f"private.step_index must be a non-negative integer: {self.private.step_index!r}")
        if not isinstance(self.private.invalid_action_count, int) or isinstance(self.private.invalid_action_count, bool) or self.private.invalid_action_count < 0:
            fail(f"private.invalid_action_count must be a non-negative integer: {self.private.invalid_action_count!r}")

        if world.width < 5 or world.height < 5 or world.levels <= 0:
            fail(f"invalid dimensions {world.width}x{world.height}x{world.levels}")
        self._assert_layer_stack("maps", world.maps, world.levels, world.width, world.height, VALID_BLOCKS, fail)
        self._assert_layer_stack("item_maps", world.item_maps, world.levels, world.width, world.height, VALID_ITEMS, fail)
        self._assert_numeric_layer_stack("light_maps", world.light_maps, world.levels, world.width, world.height, fail, minimum=0.0, maximum=1.0)

        for name, values in {
            "down_ladders": world.down_ladders,
            "up_ladders": world.up_ladders,
            "chests_opened": world.chests_opened,
            "monsters_killed": world.monsters_killed,
        }.items():
            if len(values) != world.levels:
                fail(f"{name} length={len(values)} expected={world.levels}")
        if sorted(world.potion_mapping) != list(range(6)):
            fail(f"potion_mapping must be a permutation of 0..5: {world.potion_mapping}")
        for level, flag in enumerate(world.chests_opened):
            if not isinstance(flag, bool):
                fail(f"chests_opened[{level}] must be boolean: {flag!r}")
        for level, count in enumerate(world.monsters_killed):
            if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count > 2_147_483_647:
                fail(f"monsters_killed[{level}] must be a nonnegative int32 counter: {count!r}")

        self._assert_ladder_consistency("ladder_down", world.down_ladders, fail)
        self._assert_ladder_consistency("ladder_up", world.up_ladders, fail)
        self._assert_item_map_invariants(fail)

        if not _in_bounds(world.player_pos, world.width, world.height):
            fail(f"player_pos out of bounds: {world.player_pos}")
        if not 0 <= world.player_level < world.levels:
            fail(f"player_level out of range: {world.player_level}")
        if _in_bounds(world.player_pos, world.width, world.height) and 0 <= world.player_level < world.levels:
            player_block = normalize_tile(world.maps[world.player_level][world.player_pos[1]][world.player_pos[0]])
            if not _is_land_walkable(player_block):
                fail(f"player on blocked terrain {player_block!r}: level={world.player_level} pos={world.player_pos}")
        if world.player_direction not in VALID_DIRECTIONS:
            fail(f"player_direction must be cardinal: {world.player_direction}")
        if not isinstance(world.is_sleeping, bool):
            fail(f"is_sleeping must be boolean: {world.is_sleeping!r}")
        if not isinstance(world.is_resting, bool):
            fail(f"is_resting must be boolean: {world.is_resting!r}")
        if world.is_sleeping and world.is_resting:
            fail("player cannot be sleeping and resting simultaneously")

        self._assert_inventory_invariants(fail)
        self._assert_achievement_invariants(fail)
        self._assert_entity_invariants(fail)
        self._assert_projectile_invariants("player_projectiles", world.player_projectiles, "player", fail)
        self._assert_projectile_invariants("mob_projectiles", world.mob_projectiles, "mob", fail)
        self._assert_growing_plant_invariants(fail)

        for name, value in {
            "reward_last": self.private.reward_last,
            "total_reward": self.private.total_reward,
            "player_recover": world.player_recover,
            "player_hunger": world.player_hunger,
            "player_thirst": world.player_thirst,
            "player_fatigue": world.player_fatigue,
            "player_recover_mana": world.player_recover_mana,
            "light_level": world.light_level,
        }.items():
            if not _finite_number(value):
                fail(f"{name} must be finite: {value!r}")
        for name, value, minimum, maximum in (
            ("player_recover", world.player_recover, -15.0, 25.0),
            ("player_hunger", world.player_hunger, 0.0, 25.0),
            ("player_thirst", world.player_thirst, 0.0, 20.0),
            ("player_fatigue", world.player_fatigue, -10.0, 30.0),
            ("player_recover_mana", world.player_recover_mana, 0.0, 30.0),
        ):
            if _finite_number(value) and not minimum <= float(value) <= maximum:
                fail(f"{name}={value!r} outside runtime range [{minimum}, {maximum}]")
        expected_light = _calculate_light_level(world.timestep, self._day_length())
        if abs(float(world.light_level) - expected_light) > 1e-9:
            fail(f"light_level={world.light_level} expected={expected_light} at timestep={world.timestep}")
        if self.private.terminated and self.private.truncated:
            fail("episode cannot be both terminated and truncated")
        if self.private.done_reason is not None and self.private.done_reason not in {"death", "boss_defeated", "max_steps"}:
            fail(f"unknown done_reason: {self.private.done_reason}")
        expected_done_reason = self._done_reason()
        if expected_done_reason is None:
            if self.private.terminated or self.private.truncated or self.private.done_reason is not None:
                fail(
                    "private terminal flags disagree with live state: "
                    f"terminated={self.private.terminated} truncated={self.private.truncated} done_reason={self.private.done_reason!r}"
                )
        elif expected_done_reason == "max_steps":
            if not self.private.truncated or self.private.terminated or self.private.done_reason != expected_done_reason:
                fail(
                    "private terminal flags disagree with max_steps state: "
                    f"terminated={self.private.terminated} truncated={self.private.truncated} done_reason={self.private.done_reason!r}"
                )
        elif not self.private.terminated or self.private.truncated or self.private.done_reason != expected_done_reason:
            fail(
                f"private terminal flags disagree with {expected_done_reason} state: "
                f"terminated={self.private.terminated} truncated={self.private.truncated} done_reason={self.private.done_reason!r}"
            )
        if errors:
            message = "; ".join(errors[:12])
            if len(errors) > 12:
                message += f"; ... {len(errors) - 12} more"
            raise CraftaxInvariantError(f"Craftax invariant violation at {context}: {message}")

    def _assert_layer_stack(
        self,
        name: str,
        stack: list[list[list[str]]],
        levels: int,
        width: int,
        height: int,
        valid_values: set[str],
        fail: Any,
    ) -> None:
        if len(stack) != levels:
            fail(f"{name} levels={len(stack)} expected={levels}")
            return
        for level, grid in enumerate(stack):
            if len(grid) != height:
                fail(f"{name}[{level}] rows={len(grid)} expected={height}")
                continue
            for y, row in enumerate(grid):
                if len(row) != width:
                    fail(f"{name}[{level}][{y}] columns={len(row)} expected={width}")
                    continue
                for x, value in enumerate(row):
                    if value not in valid_values:
                        fail(f"{name}[{level}][{y}][{x}] unknown value: {value!r}")
                        return

    def _assert_numeric_layer_stack(
        self,
        name: str,
        stack: list[list[list[float]]],
        levels: int,
        width: int,
        height: int,
        fail: Any,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        if len(stack) != levels:
            fail(f"{name} levels={len(stack)} expected={levels}")
            return
        for level, grid in enumerate(stack):
            if len(grid) != height:
                fail(f"{name}[{level}] rows={len(grid)} expected={height}")
                continue
            for y, row in enumerate(grid):
                if len(row) != width:
                    fail(f"{name}[{level}][{y}] columns={len(row)} expected={width}")
                    continue
                for x, value in enumerate(row):
                    if not _finite_number(value):
                        fail(f"{name}[{level}][{y}][{x}] must be finite: {value!r}")
                        return
                    numeric = float(value)
                    if minimum is not None and numeric < minimum:
                        fail(f"{name}[{level}][{y}][{x}]={numeric!r} below {minimum!r}")
                        return
                    if maximum is not None and numeric > maximum:
                        fail(f"{name}[{level}][{y}][{x}]={numeric!r} above {maximum!r}")
                        return

    def _assert_ladder_consistency(self, item: str, positions: list[tuple[int, int]], fail: Any) -> None:
        assert self.world is not None
        for level, pos in enumerate(positions):
            if pos == (-1, -1):
                continue
            if item == "ladder_up" and level in {0, self.world.levels - 1}:
                continue
            if item == "ladder_down" and level == self.world.levels - 1:
                continue
            if not _in_bounds(pos, self.world.width, self.world.height):
                fail(f"{item}[{level}] out of bounds: {pos}")
                continue
            if level < len(self.world.item_maps) and self.world.item_maps[level][pos[1]][pos[0]] != item:
                fail(f"{item}[{level}] points at {self.world.item_maps[level][pos[1]][pos[0]]!r}")

    def _assert_item_map_invariants(self, fail: Any) -> None:
        assert self.world is not None
        ladder_positions = {
            "ladder_down": list(self.world.down_ladders),
            "ladder_up": list(self.world.up_ladders),
        }
        ladder_counts = {
            "ladder_down": [0 for _ in range(self.world.levels)],
            "ladder_up": [0 for _ in range(self.world.levels)],
        }
        for level, item_map in enumerate(self.world.item_maps):
            if level >= len(self.world.maps):
                continue
            block_map = self.world.maps[level]
            for y, row in enumerate(item_map):
                if y >= len(block_map):
                    continue
                for x, item in enumerate(row):
                    if item == "none":
                        continue
                    if x >= len(block_map[y]):
                        continue
                    block = normalize_tile(block_map[y][x])
                    if item == "torch" and block not in CAN_PLACE_ITEM_ON:
                        fail(f"torch[{level}] on non-placeable block {block!r} at {(x, y)}")
                    if item in ladder_positions:
                        ladder_counts[item][level] += 1
                        if not _ladder_allowed(level, item):
                            fail(f"{item}[{level}] is not allowed at {(x, y)}")
                        positions = ladder_positions[item]
                        if level >= len(positions) or positions[level] != (x, y):
                            expected = positions[level] if level < len(positions) else None
                            fail(f"{item}[{level}] extra item at {(x, y)} expected {expected}")
        for item, counts in ladder_counts.items():
            for level, count in enumerate(counts):
                if count > 1:
                    fail(f"{item}[{level}] count {count} exceeds 1")

    def _assert_inventory_invariants(self, fail: Any) -> None:
        assert self.world is not None
        inv = self.world.inventory
        defaults = default_inventory()
        count_keys = {
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "sapling",
            "ruby",
            "sapphire",
            "bow",
            "arrows",
            "torches",
            "books",
            "xp",
            "boss_progress",
        }
        missing = sorted(set(defaults) - set(inv))
        if missing:
            fail(f"inventory missing keys: {missing}")
        for key, value in inv.items():
            if key in {"learned_spells", "sword_enchantment", "bow_enchantment", "armour_enchantments"}:
                continue
            if key == "potions":
                if set(value) != set(POTION_COLORS):
                    fail(f"potions keys mismatch: {sorted(value)}")
                for color, count in value.items():
                    if color not in POTION_COLORS or not _nonnegative_int(count):
                        fail(f"invalid potion count {color}={count!r}")
                continue
            if key == "armour":
                if not isinstance(value, list) or len(value) != 4 or any(not _int_in_range(piece, 0, 2) for piece in value):
                    fail(f"invalid armour vector: {value!r}")
                continue
            if key in count_keys and not _nonnegative_int(value):
                fail(f"inventory {key} must be a non-negative integer count: {value!r}")
                continue
            if isinstance(value, (int, float)) and not _nonnegative_finite(value):
                fail(f"inventory {key} must be non-negative finite: {value!r}")
        for tier_key in ("pickaxe", "sword"):
            if not _int_in_range(inv.get(tier_key, 0), 0, 4):
                fail(f"{tier_key} tier out of range: {inv.get(tier_key)!r}")
        for attr_key in ("dexterity", "strength", "intelligence"):
            if not _int_in_range(inv.get(attr_key, 0), 1, 5):
                fail(f"{attr_key} out of range: {inv.get(attr_key)!r}")
        if not isinstance(inv.get("learned_spells"), list) or any(str(spell) not in {"fireball", "iceball"} for spell in inv.get("learned_spells", [])):
            fail(f"invalid learned_spells: {inv.get('learned_spells')!r}")
        if inv.get("sword_enchantment") not in {"none", "fire", "ice"}:
            fail(f"invalid sword_enchantment: {inv.get('sword_enchantment')!r}")
        if inv.get("bow_enchantment") not in {"none", "fire", "ice"}:
            fail(f"invalid bow_enchantment: {inv.get('bow_enchantment')!r}")
        armour_enchantments = inv.get("armour_enchantments")
        if not isinstance(armour_enchantments, list) or len(armour_enchantments) != 4 or any(enchantment not in {"none", "fire", "ice"} for enchantment in armour_enchantments):
            fail(f"invalid armour_enchantments: {armour_enchantments!r}")
        for stat in ("health", "food", "drink", "energy", "mana"):
            value = inv.get(stat)
            minimum = 9.0 if stat == "health" and bool(self.resolved and self.resolved.rules.get("god_mode", False)) else 0.0
            if not _finite_number(value) or float(value) < minimum or float(value) > _max_stat(self.world, stat):
                fail(f"{stat} out of range: {value!r}")

    def _assert_achievement_invariants(self, fail: Any) -> None:
        assert self.world is not None
        unknown_world = sorted(set(self.world.achievements) - set(CRAFTAX_ACHIEVEMENTS))
        unknown_private = sorted(set(self.private.achievements) - set(CRAFTAX_ACHIEVEMENTS))
        if unknown_world:
            fail(f"unknown world achievements: {unknown_world}")
        if unknown_private:
            fail(f"unknown private achievements: {unknown_private}")
        for name, count in self.world.achievements.items():
            if not isinstance(count, int) or isinstance(count, bool) or count not in {0, 1}:
                fail(f"achievement {name} must be binary integer: {count!r}")
        active_world_achievements = {name for name, count in self.world.achievements.items() if count > 0}
        if active_world_achievements != self.private.achievements:
            fail(f"private achievements differ from world achievements: {sorted(self.private.achievements ^ active_world_achievements)}")

    def _assert_entity_invariants(self, fail: Any) -> None:
        assert self.world is not None
        active_positions: set[tuple[int, tuple[int, int]]] = set()
        active_ids: set[str] = set()
        for entity in self.world.entities:
            if not isinstance(entity.mask, bool):
                fail(f"entity mask must be boolean: {entity.to_dict()}")
            if entity.kind not in MOB_NAMES:
                fail(f"unknown entity kind: {entity.kind!r}")
            if (entity.mob_class or _mob_class(entity.kind)) not in {"passive", "melee", "ranged"}:
                fail(f"invalid entity class: {entity.mob_class!r}")
            if not isinstance(entity.level, int) or isinstance(entity.level, bool):
                fail(f"entity level must be integer: {entity.to_dict()}")
            if not isinstance(entity.attack_cooldown, int) or isinstance(entity.attack_cooldown, bool):
                fail(f"entity attack_cooldown must be integer: {entity.to_dict()}")
            mob_class = entity.mob_class or _mob_class(entity.kind)
            max_cooldown = {"passive": 0, "melee": 5, "ranged": 4}.get(mob_class)
            if entity.mask and mob_class == "passive" and entity.attack_cooldown != 0:
                fail(f"entity attack_cooldown={entity.attack_cooldown!r} outside passive range [0, 0]: {entity.to_dict()}")
            elif entity.mask and max_cooldown is not None and entity.attack_cooldown > max_cooldown:
                fail(f"entity attack_cooldown={entity.attack_cooldown!r} above {mob_class} max {max_cooldown}: {entity.to_dict()}")
            if len(entity.pos) != 2 or any(not isinstance(component, int) or isinstance(component, bool) for component in entity.pos):
                fail(f"entity position must be integer coordinates: {entity.to_dict()}")
            if entity.level < 0 or entity.level >= self.world.levels:
                fail(f"entity level out of range: {entity.to_dict()}")
            if not _finite_number(entity.health):
                fail(f"entity health must be finite: {entity.to_dict()}")
            if entity.mask:
                if not entity.id:
                    fail(f"active entity missing id: {entity.to_dict()}")
                if entity.id in active_ids:
                    fail(f"duplicate active entity id: {entity.id}")
                active_ids.add(entity.id)
                if entity.health <= 0:
                    fail(f"active entity has non-positive health: {entity.to_dict()}")
                max_health = float(MOB_HEALTH[entity.kind])
                if entity.health > max_health:
                    fail(f"active entity health {entity.health!r} exceeds {entity.kind} max {max_health!r}: {entity.to_dict()}")
                if not _in_bounds(entity.pos, self.world.width, self.world.height):
                    fail(f"active entity out of bounds: {entity.to_dict()}")
                else:
                    terrain_error = self._entity_terrain_error(entity)
                    if terrain_error is not None:
                        fail(terrain_error)
                position_key = (entity.level, entity.pos)
                if position_key in active_positions:
                    fail(f"multiple active entities at {position_key}")
                active_positions.add(position_key)
                if entity.level == self.world.player_level and entity.pos == self.world.player_pos:
                    fail(f"active entity overlaps player: {entity.to_dict()}")

    def _entity_terrain_error(self, entity: Entity) -> str | None:
        assert self.world is not None
        block = normalize_tile(self.world.maps[entity.level][entity.pos[1]][entity.pos[0]])
        if block in STATIC_SOLID_BLOCKS:
            return f"active entity in solid block {block!r}: {entity.to_dict()}"
        mob_class = entity.mob_class or _mob_class(entity.kind)
        if not self._mob_can_occupy_block(entity.kind, mob_class, block):
            in_water = block == "water"
            in_lava = block == "lava"
            if not in_water and not in_lava:
                return f"active entity cannot occupy ground block {block!r}: {entity.to_dict()}"
            if in_water:
                return f"active entity cannot occupy water: {entity.to_dict()}"
            return f"active entity cannot occupy lava: {entity.to_dict()}"
        return None

    def _mob_can_occupy_block(self, kind: str, mob_class: str, block: str) -> bool:
        if block in STATIC_SOLID_BLOCKS:
            return False
        if mob_class not in MOB_CLASSES:
            return True
        mob_type_id = MOB_TYPE_IDS.get(kind, 0)
        collision_map = MOB_TYPE_COLLISION_MAPPING[mob_type_id][MOB_CLASSES.index(mob_class)]
        in_water = block == "water"
        in_lava = block == "lava"
        on_ground = block not in STATIC_SOLID_BLOCKS and not in_water and not in_lava
        if collision_map[0] and on_ground:
            return False
        if collision_map[1] and in_water:
            return False
        if collision_map[2] and in_lava:
            return False
        return True

    def _assert_projectile_invariants(self, name: str, projectiles: list[Projectile], owner: str, fail: Any) -> None:
        assert self.world is not None
        active_ids: set[str] = set()
        active_by_level = {level: 0 for level in range(self.world.levels)}
        slots_by_level = {level: 0 for level in range(self.world.levels)}
        active_entity_positions = {
            (entity.level, entity.pos)
            for entity in self.world.entities
            if entity.mask
        }
        capacity_key = "max_player_projectiles" if owner == "player" else "max_mob_projectiles"
        default_capacity = 3
        capacity = int(self.resolved.world.get(capacity_key, default_capacity)) if self.resolved else default_capacity
        for projectile in projectiles:
            if not isinstance(projectile.mask, bool):
                fail(f"{name} mask must be boolean: {projectile.to_dict()}")
            if projectile.kind not in PROJECTILE_TYPES:
                fail(f"{name} unknown kind: {projectile.to_dict()}")
            if projectile.owner not in VALID_PROJECTILE_OWNERS:
                fail(f"{name} invalid owner: {projectile.to_dict()}")
            if projectile.owner != owner:
                fail(f"{name} owner mismatch: {projectile.to_dict()}")
            if not _projectile_kind_allowed_for_owner(projectile.kind, owner):
                fail(f"{name} kind {projectile.kind!r} cannot be owned by {owner}: {projectile.to_dict()}")
            if not isinstance(projectile.level, int) or isinstance(projectile.level, bool):
                fail(f"{name} level must be integer: {projectile.to_dict()}")
            if len(projectile.pos) != 2 or any(not isinstance(component, int) or isinstance(component, bool) for component in projectile.pos):
                fail(f"{name} position must be integer coordinates: {projectile.to_dict()}")
            if len(projectile.direction) != 2 or any(not isinstance(component, int) or isinstance(component, bool) for component in projectile.direction):
                fail(f"{name} direction must use integer components: {projectile.to_dict()}")
            if projectile.direction not in VALID_DIRECTIONS:
                fail(f"{name} direction must be cardinal: {projectile.to_dict()}")
            if projectile.level < 0 or projectile.level >= self.world.levels:
                fail(f"{name} level out of range: {projectile.to_dict()}")
            else:
                slots_by_level[projectile.level] = slots_by_level.get(projectile.level, 0) + 1
            if projectile.mask:
                if not projectile.id:
                    fail(f"{name} active projectile missing id: {projectile.to_dict()}")
                if projectile.id in active_ids:
                    fail(f"{name} duplicate active id: {projectile.id}")
                active_ids.add(projectile.id)
                active_by_level[projectile.level] = active_by_level.get(projectile.level, 0) + 1
                if not _in_bounds(projectile.pos, self.world.width, self.world.height):
                    fail(f"{name} active projectile out of bounds: {projectile.to_dict()}")
                else:
                    block = normalize_tile(self.world.maps[projectile.level][projectile.pos[1]][projectile.pos[0]])
                    if block in STATIC_SOLID_BLOCKS:
                        fail(f"{name} active projectile in solid block {block!r}: {projectile.to_dict()}")
                    if projectile.level == self.world.player_level and projectile.pos == self.world.player_pos:
                        fail(f"{name} active projectile overlaps player: {projectile.to_dict()}")
                    if (projectile.level, projectile.pos) in active_entity_positions:
                        fail(f"{name} active projectile overlaps entity: {projectile.to_dict()}")
        for level, count in active_by_level.items():
            if count > capacity:
                fail(f"{name}[{level}] active count {count} exceeds capacity {capacity}")
        for level, count in slots_by_level.items():
            if count > capacity:
                fail(f"{name}[{level}] slot count {count} exceeds capacity {capacity}")

    def _assert_growing_plant_invariants(self, fail: Any) -> None:
        assert self.world is not None
        if len(self.world.growing_plants) > MAX_GROWING_PLANTS:
            fail(f"growing plant slot count {len(self.world.growing_plants)} exceeds {MAX_GROWING_PLANTS}")
        active_count = 0
        for index, plant in enumerate(self.world.growing_plants):
            active = plant.get("active", False)
            if not isinstance(active, bool):
                fail(f"growing_plants[{index}].active must be boolean: {active!r}")
                continue
            if active:
                active_count += 1
        if active_count > MAX_GROWING_PLANTS:
            fail(f"active growing plant count {active_count} exceeds {MAX_GROWING_PLANTS}")
        for index, plant in enumerate(self.world.growing_plants):
            pos = _pos(plant.get("pos", (-1, -1)))
            active = plant.get("active", False)
            if isinstance(active, bool) and active and not _in_bounds(pos, self.world.width, self.world.height):
                fail(f"active growing plant out of bounds: {plant!r}")
            if not _nonnegative_finite(plant.get("age", 0)):
                fail(f"invalid growing plant age: {plant!r}")


def normalize_action(raw: str | dict[str, Any]) -> str:
    if isinstance(raw, dict):
        value = str(raw.get("action") or raw.get("kind") or raw.get("type") or "noop")
    else:
        value = str(raw)
        if value.strip().startswith("{"):
            return normalize_action(json.loads(value))
    return ACTION_ALIASES.get(value, value)


def normalize_tile(kind: str) -> str:
    tile = TILE_ALIASES.get(str(kind), str(kind))
    return tile if tile in set(BLOCK_TYPES) | {"torch", "ladder_down", "ladder_up", "ladder_down_blocked"} else tile


def _day_length_from_resolved(resolved: ResolvedTask) -> int:
    value = resolved.world.get("day_length", resolved.rules.get("day_length", DAY_LENGTH))
    return max(1, int(value))


def _ladder_allowed(level: int, item: str) -> bool:
    kind, config = LEVEL_STACK[min(max(level, 0), len(LEVEL_STACK) - 1)]
    if kind == "dungeon":
        return True
    attr = "ladder_down" if item == "ladder_down" else "ladder_up"
    return bool(getattr(config, attr, False))


def _calculate_light_level(timestep: int, day_length: int) -> float:
    progress = (int(timestep) / max(1, int(day_length))) % 1 + 0.3
    return 1.0 - abs(math.cos(math.pi * progress)) ** 3


def _is_land_walkable(tile: str) -> bool:
    return tile not in STATIC_SOLID_BLOCKS and tile not in {"water", "lava"}


def normalize_item(kind: str) -> str:
    value = str(kind)
    if value in ITEM_OVERLAYS:
        return value
    if value == "none":
        return "none"
    return "none"


def normalize_mob(kind: str) -> str:
    return MOB_ALIASES.get(str(kind), str(kind))


def _mob_class(kind: str) -> str:
    if kind in PASSIVE_MOBS:
        return "passive"
    if kind in MELEE_MOBS:
        return "melee"
    if kind in RANGED_MOBS:
        return "ranged"
    return "melee"


def normalize_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    normalized = _deep_merge(default_inventory(), inventory)
    armour = normalized.get("armour", [0, 0, 0, 0])
    if isinstance(armour, int) and not isinstance(armour, bool):
        normalized["armour"] = [armour] * 4
    elif isinstance(armour, (list, tuple)):
        pieces = list(armour)[:4]
        normalized["armour"] = pieces + [0] * (4 - len(pieces))
    else:
        normalized["armour"] = armour
    enchants = normalized.get("armour_enchantments", ["none", "none", "none", "none"])
    if isinstance(enchants, str):
        normalized["armour_enchantments"] = [enchants] * 4
    else:
        values = [str(value) for value in list(enchants)[:4]]
        normalized["armour_enchantments"] = values + ["none"] * (4 - len(values))
    normalized.pop("armour_enchantment", None)
    normalized["learned_spells"] = list(dict.fromkeys(str(spell) for spell in normalized.get("learned_spells", [])))
    normalized["potions"] = _deep_merge(default_inventory()["potions"], normalized.get("potions", {}))
    return normalized


def _resource_replacement(tile: str, level: int) -> str:
    if tile == "tree":
        return "grass"
    if tile == "fire_tree":
        return "fire_grass"
    if tile == "ice_shrub":
        return "ice_grass"
    return "path" if level > 0 else "grass"


def _max_stat(world: CraftaxWorld, stat: str) -> int:
    if stat == "health":
        return 8 + int(world.inventory.get("strength", 1))
    if stat in {"food", "drink", "energy"}:
        return 7 + 2 * int(world.inventory.get("dexterity", 1))
    if stat == "mana":
        return 6 + 3 * int(world.inventory.get("intelligence", 1))
    return 9


def _armour(world: CraftaxWorld) -> list[int]:
    armour = world.inventory.get("armour", [0, 0, 0, 0])
    if isinstance(armour, int):
        return [int(armour)] * 4
    pieces = [int(piece) for piece in list(armour)[:4]]
    return pieces + [0] * (4 - len(pieces))


def observation_text_from_readout(observation: dict[str, Any], valid_actions: list[str]) -> str:
    inv = observation["inventory"]
    lines = [
        f"level: {observation['player']['level']}",
        f"position: {observation['player']['pos']} direction={observation['player']['direction']}",
        f"front_tile: {observation['player']['front_tile']}",
        "local_map:",
        *observation["local_map"],
        "inventory: " + ", ".join(f"{key}={value}" for key, value in inv.items() if key not in {"potions", "learned_spells"}),
        "potions: " + ", ".join(f"{key}={value}" for key, value in inv.get("potions", {}).items()),
        "learned_spells: " + ", ".join(inv.get("learned_spells", [])),
        "achievements: " + ", ".join(observation["achievements"]),
        "nearby_entities: " + ", ".join(f"{e['kind']}@{e['pos']} hp={e['health']}" for e in observation["nearby_entities"]),
        "projectiles: " + ", ".join(
            f"{p['owner']}:{p['kind']}@{p['pos']} dir={p['direction']}"
            for group in observation.get("projectile_state", {}).values()
            for p in group
        ),
        "valid_actions: " + ", ".join(valid_actions),
    ]
    return "\n".join(lines)


def _resolved_from_dict(data: dict[str, Any]) -> ResolvedTask:
    seed = _strict_int(data["seed"], "resolved.seed")
    width = _strict_int(data["width"], "resolved.width")
    height = _strict_int(data["height"], "resolved.height")
    max_steps = _strict_int(data["max_steps"], "resolved.max_steps")
    if width < 5 or height < 5:
        raise CraftaxInvariantError(f"resolved dimensions must be at least 5x5: {width}x{height}")
    if max_steps <= 0:
        raise CraftaxInvariantError(f"resolved.max_steps must be positive: {max_steps!r}")
    task_id = str(data["task_id"])
    scenario_id = str(data["scenario_id"])
    world = copy.deepcopy(data["world"])
    rules = copy.deepcopy(data["rules"])
    readouts = copy.deepcopy(data.get("readouts", {}))
    for field, actual, expected in (
        ("world.seed", world.get("seed"), seed),
        ("world.width", world.get("width"), width),
        ("world.height", world.get("height"), height),
        ("world.max_steps", world.get("max_steps"), max_steps),
    ):
        if actual != expected:
            raise CraftaxInvariantError(f"resolved {field}={actual!r} does not match resolved projection {expected!r}")
    expanded = {
        "task_id": task_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "world": world,
        "rules": rules,
        "readouts": readouts,
    }
    config_hash = str(data["config_hash"])
    recomputed_hash = stable_hash(expanded, 16)
    if config_hash != recomputed_hash:
        raise CraftaxInvariantError(f"resolved.config_hash={config_hash!r} recomputed={recomputed_hash!r}")
    episode_id = str(data["episode_id"])
    recomputed_episode_id = stable_hash(f"gamebench.craftax-singleplayer.episode:{task_id}:{seed}:{config_hash}", 32)
    if episode_id != recomputed_episode_id:
        raise CraftaxInvariantError(f"resolved.episode_id={episode_id!r} recomputed={recomputed_episode_id!r}")
    return ResolvedTask(
        task_id=task_id,
        scenario_id=scenario_id,
        seed=seed,
        width=width,
        height=height,
        max_steps=max_steps,
        world=world,
        rules=rules,
        readouts=readouts,
        config_hash=config_hash,
        episode_id=episode_id,
    )


def _json_to_rng_state(value: Any) -> object:
    if isinstance(value, list):
        return tuple(_json_to_rng_state(item) for item in value)
    return value


def _normalize_item_maps(value: Any, maps: list[list[list[str]]]) -> list[list[list[str]]]:
    if value is not None:
        return [[[normalize_item(cell) for cell in row] for row in level] for level in value]
    item_maps = [[["none" for _ in row] for row in level] for level in maps]
    for level_index, level in enumerate(maps):
        for y, row in enumerate(level):
            for x, cell in enumerate(row):
                normalized = normalize_tile(cell)
                if normalized in ITEM_OVERLAYS:
                    item_maps[level_index][y][x] = normalized
                    maps[level_index][y][x] = _default_floor_tile(level_index)
                else:
                    maps[level_index][y][x] = normalized
    return item_maps


def _normalize_light_maps(value: Any, levels: int, width: int, height: int) -> list[list[list[float]]]:
    if value is not None:
        return [[[_strict_finite_number(cell, f"light_maps[{level_idx}][{row_idx}][{cell_idx}]") for cell_idx, cell in enumerate(row)] for row_idx, row in enumerate(level)] for level_idx, level in enumerate(value)]
    return [[[0.0 for _ in range(width)] for _ in range(height)] for _ in range(levels)]


def _normalize_positions(value: Any, levels: int, default: tuple[int, int]) -> list[tuple[int, int]]:
    if value is None:
        return [default for _ in range(levels)]
    positions = [_pos(item) for item in list(value)[:levels]]
    return positions + [default for _ in range(levels - len(positions))]


def _normalize_int_vector(value: Any, levels: int, field: str) -> list[int]:
    if value is None:
        return [0 for _ in range(levels)]
    if isinstance(value, dict):
        out = [0 for _ in range(levels)]
        for key, count in value.items():
            out[int(key)] = _strict_int(count, f"{field}[{key}]")
        return out
    items = [_strict_int(item, f"{field}[{idx}]") for idx, item in enumerate(list(value)[:levels])]
    return items + [0 for _ in range(levels - len(items))]


def _normalize_bool_vector(value: Any, levels: int, field: str) -> list[bool]:
    if value is None:
        return [False for _ in range(levels)]
    if isinstance(value, dict):
        out = [False for _ in range(levels)]
        for key, flag in value.items():
            out[int(key)] = _strict_bool(flag, f"{field}[{key}]")
        return out
    items = [_strict_bool(item, f"{field}[{idx}]") for idx, item in enumerate(list(value)[:levels])]
    return items + [False for _ in range(levels - len(items))]


def _strict_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    raise CraftaxInvariantError(f"{field} must be boolean: {value!r}")


def _projectiles_from_initial(items: list[dict[str, Any]], owner: str, default_level: int) -> list[Projectile]:
    projectiles: list[Projectile] = []
    for idx, item in enumerate(items):
        kind = str(item["kind"])
        projectiles.append(
            Projectile(
                id=str(item.get("id", f"{owner}_projectile_{idx}")),
                kind=kind,
                owner=str(item.get("owner", owner)),
                level=_strict_int(item.get("level", default_level), f"{owner}_projectile.level"),
                pos=_pos(item["pos"], f"{owner}_projectile.pos"),
                direction=_dir(item.get("direction", (0, 1)), f"{owner}_projectile.direction"),
                mask=_strict_bool(item.get("mask", True), f"{owner}_projectile.mask"),
            )
        )
    return projectiles


def _stable_hash(value: Any) -> str:
    import hashlib

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise CraftaxInvariantError(f"{field} must be integer: {value!r}")


def _strict_finite_number(value: Any, field: str) -> float:
    if _finite_number(value):
        return float(value)
    raise CraftaxInvariantError(f"{field} must be finite number: {value!r}")


def _pos(value: Any, field: str = "pos") -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CraftaxInvariantError(f"{field} must be a two-element coordinate: {value!r}")
    return (_strict_int(value[0], f"{field}[0]"), _strict_int(value[1], f"{field}[1]"))


def _dir(value: Any, field: str = "direction") -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CraftaxInvariantError(f"{field} must be a two-element direction: {value!r}")
    return (_strict_int(value[0], f"{field}[0]"), _strict_int(value[1], f"{field}[1]"))


def _in_bounds(pos: tuple[int, int], width: int, height: int) -> bool:
    return 0 <= pos[0] < width and 0 <= pos[1] < height


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _projectile_kind_allowed_for_owner(kind: str, owner: str) -> bool:
    if owner == "player":
        return kind in PLAYER_PROJECTILE_KINDS
    if owner == "mob":
        return kind in MOB_PROJECTILE_KINDS
    return False


def _nonnegative_finite(value: Any) -> bool:
    return _finite_number(value) and float(value) >= 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _int_in_range(value: Any, minimum: int, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and minimum <= value <= maximum


def _add_pos(pos: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return (pos[0] + direction[0], pos[1] + direction[1])


def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _euclidean(left: tuple[int, int], right: tuple[int, int]) -> float:
    return math.sqrt((left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2)


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _potion_index(color: str) -> int:
    colors = ["red", "green", "blue", "pink", "cyan", "yellow"]
    return colors.index(color)


def _find_tile(grid: list[list[str]], tile: str) -> tuple[int, int] | None:
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == tile:
                return (x, y)
    return None


def _find_item(grid: list[list[str]], item: str) -> tuple[int, int] | None:
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if normalize_item(value) == item:
                return (x, y)
    return None


def _valid_ladder_pos(pos: tuple[int, int]) -> tuple[int, int] | None:
    return None if pos == (-1, -1) else pos


def _default_floor_tile(level: int) -> str:
    if level == 0:
        return "grass"
    if level == 6:
        return "fire_grass"
    if level == 7:
        return "ice_grass"
    return "path"


def _craft_tier(action: str) -> int:
    if "diamond" in action:
        return 4
    if "iron" in action:
        return 3
    if "stone" in action:
        return 2
    return 1


# One glyph per world feature, mirroring gold_rust/src/native.rs.  Kept
# injective: mob glyphs used to be ``kind[0].upper()``, which rendered a pigman
# as ``P`` (the player), a troll as ``T`` (a tree), and a lizard as ``L``
# (lava).
TILE_GLYPHS: dict[str, str] = {
    "grass": ".",
    "path": "_",
    "sand": ",",
    "gravel": ";",
    "fire_grass": '"',
    "ice_grass": "'",
    "water": "~",
    "fountain": "u",
    "stone": "o",
    "stalagmite": "^",
    "wall": "#",
    "wall_moss": "%",
    "tree": "T",
    "fire_tree": "Y",
    "ice_shrub": "y",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "sapphire": "s",
    "ruby": "r",
    "chest": "h",
    "crafting_table": "a",
    "furnace": "F",
    "ladder_down": ">",
    "ladder_up": "<",
    "ladder_down_blocked": "x",
    "plant": "p",
    "ripe_plant": "R",
    "torch": "t",
    "grave": "+",
    "grave2": "+",
    "grave3": "+",
    "enchantment_table_fire": "E",
    "enchantment_table_ice": "e",
    "necromancer": "N",
    "necromancer_vulnerable": "n",
    "darkness": " ",
    "out_of_bounds": " ",
    "invalid": "?",
}

ENTITY_GLYPHS: dict[str, str] = {
    "cow": "C",
    "zombie": "Z",
    "skeleton": "S",
    "bat": "B",
    "snail": "U",
    "orc_solider": "O",
    "orc_mage": "Q",
    "gnome_warrior": "G",
    "gnome_archer": "g",
    "lizard": "V",
    "kobold": "K",
    "knight": "W",
    "archer": "A",
    "troll": "M",
    "deep_thing": "D",
    "pigman": "X",
    "fire_elemental": "H",
    "frost_troll": "I",
    "ice_elemental": "J",
    "necromancer": "N",
}

PLAYER_GLYPH = "P"
UNKNOWN_GLYPH = "?"


def _tile_char(tile: str) -> str:
    return TILE_GLYPHS.get(tile, UNKNOWN_GLYPH)


def _entity_char(kind: str) -> str:
    return ENTITY_GLYPHS.get(kind, UNKNOWN_GLYPH)


def _projectile_char(projectile: Projectile) -> str:
    if projectile.owner == "mob":
        return "!"
    if projectile.kind in {"fireball", "fireball2"}:
        return "*"
    if projectile.kind in {"iceball", "iceball2"}:
        return ":"
    return "-"


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _title_event(kind: str) -> str:
    return "".join(part.capitalize() for part in kind.split("_"))
