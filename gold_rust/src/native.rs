use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

const ENV_FAMILY: &str = "craftax-singleplayer";
const TASK_SCHEMA: &str = "gamebench.task.craftax.v1";
const READOUT_SCHEMA: &str = "gamebench.craftax.readout.v1";
const MT_N: usize = 624;
const MT_M: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const POTION_COLORS: &[&str] = &["red", "green", "blue", "pink", "cyan", "yellow"];
const ACTION_NAMES: &[&str] = &[
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
    "drink_potion_red",
    "drink_potion_green",
    "drink_potion_blue",
    "drink_potion_pink",
    "drink_potion_cyan",
    "drink_potion_yellow",
    "read_book",
    "enchant_sword",
    "enchant_armour",
    "make_torch",
    "level_up_dexterity",
    "level_up_strength",
    "level_up_intelligence",
    "enchant_bow",
];

const SOLID_BLOCKS: &[&str] = &[
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
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTask {
    pub task_id: String,
    pub scenario_id: String,
    pub seed: i64,
    pub width: i64,
    pub height: i64,
    pub max_steps: i64,
    pub world: Value,
    pub rules: Value,
    pub readouts: Value,
    pub config_hash: String,
    pub episode_id: String,
}

impl ResolvedTask {
    pub fn to_value(&self) -> Value {
        json!({
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "max_steps": self.max_steps,
            "world": self.world,
            "rules": self.rules,
            "readouts": self.readouts,
            "config_hash": self.config_hash,
            "episode_id": self.episode_id,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub step_index: i64,
    pub tick: i64,
    pub episode_id: String,
    pub kind: String,
    pub action: Option<Value>,
    pub transition: Option<String>,
    pub severity: String,
    pub message: String,
    pub payload: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) pos: (i64, i64),
    pub(crate) health: f64,
    pub(crate) level: i64,
    pub(crate) mob_class: String,
    pub(crate) attack_cooldown: i64,
    pub(crate) mask: bool,
}

impl Entity {
    fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "kind": self.kind,
            "class": self.mob_class,
            "level": self.level,
            "pos": [self.pos.0, self.pos.1],
            "health": self.health,
            "attack_cooldown": self.attack_cooldown,
            "mask": self.mask,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projectile {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) pos: (i64, i64),
    pub(crate) direction: (i64, i64),
    pub(crate) level: i64,
    pub(crate) owner: String,
    pub(crate) mask: bool,
}

impl Projectile {
    fn to_value(&self) -> Value {
        json!({
            "id": self.id,
            "kind": self.kind,
            "owner": self.owner,
            "level": self.level,
            "pos": [self.pos.0, self.pos.1],
            "direction": [self.direction.0, self.direction.1],
            "mask": self.mask,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CraftaxWorld {
    pub(crate) width: i64,
    pub(crate) height: i64,
    pub(crate) levels: i64,
    pub(crate) max_steps: i64,
    pub(crate) seed: i64,
    pub(crate) maps: Vec<Vec<Vec<String>>>,
    pub(crate) item_maps: Vec<Vec<Vec<String>>>,
    pub(crate) down_ladders: Vec<(i64, i64)>,
    pub(crate) up_ladders: Vec<(i64, i64)>,
    pub(crate) chests_opened: Vec<bool>,
    pub(crate) monsters_killed: Vec<i64>,
    pub(crate) potion_mapping: Vec<i64>,
    pub(crate) player_pos: (i64, i64),
    pub(crate) player_direction: (i64, i64),
    pub(crate) player_level: i64,
    pub(crate) timestep: i64,
    pub(crate) light_level: f64,
    #[serde(default)]
    pub(crate) is_sleeping: bool,
    #[serde(default)]
    pub(crate) is_resting: bool,
    #[serde(default)]
    pub(crate) player_recover: f64,
    #[serde(default)]
    pub(crate) player_hunger: f64,
    #[serde(default)]
    pub(crate) player_thirst: f64,
    #[serde(default)]
    pub(crate) player_fatigue: f64,
    #[serde(default)]
    pub(crate) player_recover_mana: f64,
    pub(crate) inventory: Value,
    pub(crate) achievements: Value,
    pub(crate) entities: Vec<Entity>,
    pub(crate) player_projectiles: Vec<Projectile>,
    pub(crate) mob_projectiles: Vec<Projectile>,
}

impl CraftaxWorld {
    fn public_value(&self, done: bool) -> Value {
        json!({
            "observation": {},
            "player_pos": [self.player_pos.0, self.player_pos.1],
            "level": self.player_level,
            "inventory": self.inventory,
            "achievements": self.achievements,
            "done": done,
        })
    }

    fn grid_hash(&self) -> Result<String> {
        let level = usize::try_from(self.player_level).context("player_level must fit usize")?;
        let entities: Vec<Value> = self
            .entities
            .iter()
            .filter(|entity| entity.mask)
            .map(Entity::to_value)
            .collect();
        let player_projectiles: Vec<Value> = self
            .player_projectiles
            .iter()
            .filter(|projectile| projectile.mask)
            .map(Projectile::to_value)
            .collect();
        let mob_projectiles: Vec<Value> = self
            .mob_projectiles
            .iter()
            .filter(|projectile| projectile.mask)
            .map(Projectile::to_value)
            .collect();
        stable_hash(
            &json!({
                "level": self.player_level,
                "map": self.maps[level],
                "item_map": self.item_maps[level],
                "monsters_killed": self.monsters_killed,
                "entities": entities,
                "player_projectiles": player_projectiles,
                "mob_projectiles": mob_projectiles,
            }),
            16,
        )
    }

    fn ascii_map(&self) -> String {
        let level = usize::try_from(self.player_level).unwrap_or(0);
        let mut rows = Vec::new();
        for y in 0..self.height {
            let mut row = String::new();
            for x in 0..self.width {
                if (x, y) == self.player_pos {
                    row.push('P');
                    continue;
                }
                if let Some(entity) = self.entities.iter().find(|entity| {
                    entity.mask && entity.level == self.player_level && entity.pos == (x, y)
                }) {
                    row.push(entity_char(&entity.kind));
                    continue;
                }
                if let Some(projectile) = self
                    .player_projectiles
                    .iter()
                    .chain(self.mob_projectiles.iter())
                    .find(|projectile| {
                        projectile.mask
                            && projectile.level == self.player_level
                            && projectile.pos == (x, y)
                    })
                {
                    row.push(projectile_char(&projectile.kind, &projectile.owner));
                    continue;
                }
                let item = &self.item_maps[level][usize::try_from(y).unwrap()]
                    [usize::try_from(x).unwrap()];
                let tile = if item != "none" {
                    item
                } else {
                    &self.maps[level][usize::try_from(y).unwrap()][usize::try_from(x).unwrap()]
                };
                row.push(tile_char(tile));
            }
            rows.push(row);
        }
        rows.join("\n")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct CraftaxRustSession {
    pub resolved: ResolvedTask,
    #[serde(skip_serializing)]
    pub world: CraftaxWorld,
    #[serde(skip_serializing)]
    rng: PythonRandom,
    pub events: Vec<EventRecord>,
    pub private: Value,
    pub public: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointEnvelope {
    schema_version: String,
    env_family: String,
    lane: String,
    resolved: ResolvedTask,
    world: CraftaxWorld,
    rng: PythonRandomState,
    events: Vec<EventRecord>,
    private: Value,
    public: Value,
}

impl CraftaxRustSession {
    pub fn reset_from_entry(entry: &Value) -> Result<Self> {
        let task = scenario_to_task(entry)?;
        let seed_override = entry
            .get("seed")
            .map(|value| strict_int(value, "entry.seed"))
            .transpose()?;
        let resolved = resolve_task(&task, seed_override)?;
        let mut rng = PythonRandom::seed_from_i64(resolved.seed);
        let world = make_world(&resolved, &mut rng)?;
        let private = json!({
            "episode_id": resolved.episode_id,
            "task_id": resolved.task_id,
            "scenario_id": resolved.scenario_id,
            "seed": resolved.seed,
            "config_hash": resolved.config_hash,
            "step_index": 0,
            "reward_last": 0.0,
            "total_reward": 0.0,
            "terminated": false,
            "truncated": false,
            "done_reason": Value::Null,
            "achievements": [],
            "invalid_action_count": 0,
        });
        let public = world.public_value(false);
        let event = EventRecord {
            step_index: 0,
            tick: 0,
            episode_id: resolved.episode_id.clone(),
            kind: "task_resolved".to_string(),
            action: None,
            transition: Some("reset".to_string()),
            severity: "info".to_string(),
            message: format!(
                "TaskResolved({},{})",
                resolved.scenario_id, resolved.config_hash
            ),
            payload: json!({"resolved": resolved.to_value()}),
        };
        Ok(Self {
            resolved,
            world,
            rng,
            events: vec![event],
            private,
            public,
        })
    }

    pub fn reset_from_task(task: &Value, seed_override: Option<i64>) -> Result<Self> {
        let resolved = resolve_task(task, seed_override)?;
        Self::reset_from_resolved(resolved)
    }

    fn reset_from_resolved(resolved: ResolvedTask) -> Result<Self> {
        let mut rng = PythonRandom::seed_from_i64(resolved.seed);
        let world = make_world(&resolved, &mut rng)?;
        let private = json!({
            "episode_id": resolved.episode_id,
            "task_id": resolved.task_id,
            "scenario_id": resolved.scenario_id,
            "seed": resolved.seed,
            "config_hash": resolved.config_hash,
            "step_index": 0,
            "reward_last": 0.0,
            "total_reward": 0.0,
            "terminated": false,
            "truncated": false,
            "done_reason": Value::Null,
            "achievements": [],
            "invalid_action_count": 0,
        });
        let public = world.public_value(false);
        let event = EventRecord {
            step_index: 0,
            tick: 0,
            episode_id: resolved.episode_id.clone(),
            kind: "task_resolved".to_string(),
            action: None,
            transition: Some("reset".to_string()),
            severity: "info".to_string(),
            message: format!(
                "TaskResolved({},{})",
                resolved.scenario_id, resolved.config_hash
            ),
            payload: json!({"resolved": resolved.to_value()}),
        };
        Ok(Self {
            resolved,
            world,
            rng,
            events: vec![event],
            private,
            public,
        })
    }

    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>> {
        let envelope = CheckpointEnvelope {
            schema_version: "gamebench.checkpoint.v1".to_string(),
            env_family: ENV_FAMILY.to_string(),
            lane: "rust".to_string(),
            resolved: self.resolved.clone(),
            world: self.world.clone(),
            rng: self.rng.to_state(),
            events: self.events.clone(),
            private: self.private.clone(),
            public: self.public.clone(),
        };
        serde_json::to_vec(&envelope).context("encode Craftax Rust checkpoint")
    }

    pub fn restore_checkpoint_bytes(&mut self, blob: &[u8]) -> Result<usize> {
        let envelope: CheckpointEnvelope =
            serde_json::from_slice(blob).context("decode Craftax Rust checkpoint")?;
        if envelope.schema_version != "gamebench.checkpoint.v1" {
            bail!("unsupported checkpoint schema: {}", envelope.schema_version);
        }
        if envelope.env_family != ENV_FAMILY {
            bail!("unsupported checkpoint env_family: {}", envelope.env_family);
        }
        self.resolved = envelope.resolved;
        self.world = envelope.world;
        self.rng = PythonRandom::from_state(envelope.rng)?;
        self.events = envelope.events;
        self.private = envelope.private;
        self.public = envelope.public;
        Ok(self.events.len())
    }

    pub fn is_done(&self) -> bool {
        self.private
            .get("terminated")
            .and_then(Value::as_bool)
            .unwrap_or(false)
            || self
                .private
                .get("truncated")
                .and_then(Value::as_bool)
                .unwrap_or(false)
    }

    pub fn legacy_strings(&self) -> Vec<String> {
        self.events
            .iter()
            .map(|event| event.message.clone())
            .collect()
    }

    pub fn step(&mut self, raw_action: &Value) -> Result<()> {
        if self.is_done() {
            bail!("cannot step a terminal Craftax episode");
        }
        let requested_action = normalize_action_value(raw_action)?;
        let homeostasis = self
            .resolved
            .rules
            .get("homeostasis")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let continuing_sleep =
            self.world.is_sleeping && matches!(requested_action.as_str(), "noop" | "sleep");
        let continuing_rest =
            self.world.is_resting && matches!(requested_action.as_str(), "noop" | "rest");
        let waking = homeostasis
            && (self.world.is_sleeping || self.world.is_resting)
            && !continuing_sleep
            && !continuing_rest;
        let action = if homeostasis && (continuing_sleep || continuing_rest || waking) {
            "noop".to_string()
        } else {
            requested_action.clone()
        };
        if waking {
            self.world.is_sleeping = false;
            self.world.is_resting = false;
        }
        let before_inventory = self.world.inventory.clone();
        let before_achievements = self.world.achievements.clone();
        let before_health = self.inventory_f64("health")?;
        let before_pos = self.world.player_pos;
        let before_level = self.world.player_level;
        let before_tile = self.tile_at(self.world.player_pos);

        self.world.timestep += 1;
        self.set_private_i64("step_index", self.world.timestep)?;
        self.set_private_f64("reward_last", 0.0)?;

        match action.as_str() {
            "left" | "right" | "up" | "down" => self.apply_move(&action)?,
            "do" => self.apply_do()?,
            "sleep" | "rest" if homeostasis => self.append_action(
                &action,
                "intrinsic_mode_request",
                json!({"sleeping": self.world.is_sleeping, "resting": self.world.is_resting}),
            ),
            "sleep" | "rest" => self.apply_recover(&action)?,
            "read_book" => self.apply_read_book()?,
            "cast_spell" => self.apply_cast_spell()?,
            "shoot_arrow" => self.apply_shoot_arrow()?,
            "descend" => self.apply_descend()?,
            "ascend" => self.apply_ascend()?,
            action if action.starts_with("drink_potion_") => self.apply_drink_potion(action)?,
            "enchant_sword" | "enchant_armour" | "enchant_bow" => self.apply_enchant(&action)?,
            action if action.starts_with("level_up_") => self.apply_level_up(action)?,
            action if action.starts_with("place_") => self.apply_place(action)?,
            action if action.starts_with("make_") => self.apply_craft(action)?,
            "noop" => self.append_action("noop", "noop", json!({"noop": true})),
            _ => self.reject(&action, "unknown_action")?,
        }

        if self.inventory_i64("boss_progress")? >= self.world.levels - 1 {
            self.unlock("defeat_necromancer")?;
        }
        self.update_mobs(&action)?;
        self.update_player_projectiles(&action)?;
        self.spawn_mobs(&action)?;
        if homeostasis {
            self.update_intrinsics(&action)?;
        }
        self.calculate_inventory_achievements()?;
        self.apply_health_reward(before_health)?;
        self.world.light_level = calculate_light_level(self.world.timestep, self.day_length());
        self.public = self.world.public_value(false);
        let after_inventory = self.world.inventory.clone();
        self.append_inventory_deltas(&action, &before_inventory, &after_inventory)?;
        for achievement in self.newly_unlocked(&before_achievements) {
            self.add_private_achievement(&achievement)?;
            self.append_achievement(&action, &achievement);
        }
        if before_pos != self.world.player_pos || before_level != self.world.player_level {
            self.events.push(EventRecord {
                step_index: self.world.timestep,
                tick: self.world.timestep,
                episode_id: self.resolved.episode_id.clone(),
                kind: "state_transition".to_string(),
                action: Some(Value::String(action.clone())),
                transition: Some("pose".to_string()),
                severity: "info".to_string(),
                message: format!(
                    "StateTransition(level={},pos=[{}, {}])",
                    self.world.player_level, self.world.player_pos.0, self.world.player_pos.1
                ),
                payload: json!({
                    "from": {"level": before_level, "pos": [before_pos.0, before_pos.1], "tile": before_tile},
                    "to": {"level": self.world.player_level, "pos": [self.world.player_pos.0, self.world.player_pos.1], "tile": self.tile_at(self.world.player_pos)},
                }),
            });
        }
        let reward_last = self.private_f64("reward_last")?;
        if reward_last != 0.0 {
            self.events.push(EventRecord {
                step_index: self.world.timestep,
                tick: self.world.timestep,
                episode_id: self.resolved.episode_id.clone(),
                kind: "reward_delta".to_string(),
                action: Some(Value::String(action.clone())),
                transition: Some("reward".to_string()),
                severity: "info".to_string(),
                message: format!(
                    "RewardDelta({:.2},total={:.2})",
                    reward_last,
                    self.private_f64("total_reward")?
                ),
                payload: json!({"reward": reward_last, "total_reward": self.private_f64("total_reward")?}),
            });
        }
        let done_reason = if self.inventory_f64("health")? <= 0.0 {
            Some("death")
        } else if self
            .world
            .achievements
            .get("defeat_necromancer")
            .and_then(Value::as_i64)
            .unwrap_or(0)
            > 0
        {
            Some("boss_defeated")
        } else if self.world.timestep >= self.resolved.max_steps {
            Some("max_steps")
        } else {
            None
        };
        if let Some(reason) = done_reason {
            set_key(
                &mut self.private,
                "terminated",
                json!(matches!(reason, "death" | "boss_defeated")),
            )?;
            set_key(&mut self.private, "truncated", json!(reason == "max_steps"))?;
            set_key(&mut self.private, "done_reason", json!(reason))?;
            self.public = self.world.public_value(true);
            if reason == "death" {
                self.append_terminal_event("death", "Death(death)", reason);
            } else if reason == "max_steps" {
                self.append_terminal_event(
                    "episode_truncated",
                    "EpisodeTruncated(max_steps)",
                    reason,
                );
            } else {
                self.append_terminal_event(
                    "terminal_success",
                    "TerminalSuccess(boss_defeated)",
                    reason,
                );
            }
            self.append_terminal_event("terminal", &format!("Terminal({reason})"), reason);
        }
        let interval = self
            .resolved
            .world
            .get("checkpoint_every_n_steps")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if interval > 0 && self.world.timestep > 0 && self.world.timestep % interval == 0 {
            self.events.push(EventRecord {
                step_index: self.world.timestep,
                tick: self.world.timestep,
                episode_id: self.resolved.episode_id.clone(),
                kind: "checkpoint_cadence".to_string(),
                action: None,
                transition: Some("cadence".to_string()),
                severity: "info".to_string(),
                message: format!("CheckpointCadence(step={})", self.world.timestep),
                payload: json!({"step_index": self.world.timestep, "nev_cursor": self.events.len()}),
            });
        }
        Ok(())
    }

    pub fn readout(&self) -> Value {
        let observation = self.observation();
        let valid_actions = self.valid_actions();
        let observation_text = observation_text(&observation);
        json!({
            "schema": READOUT_SCHEMA,
            "env_family": ENV_FAMILY,
            "task_id": self.resolved.task_id,
            "scenario_id": self.resolved.scenario_id,
            "public": self.public,
            "private": self.private,
            "observation": observation,
            "observation_text": observation_text,
            "ascii": self.world.ascii_map(),
            "grid_hash": self.world.grid_hash().expect("grid hash should encode"),
            "nev_cursor": self.events.len(),
            "valid_actions": valid_actions,
        })
    }

    pub fn policy_readout(&self) -> Value {
        let observation = self.observation();
        json!({
            "observation": observation,
            "observation_text": "",
            "ascii": self.world.ascii_map(),
        })
    }

    pub fn scenario_output(&self) -> Value {
        json!({
            "scenario_id": self.resolved.scenario_id,
            "events": self.legacy_strings(),
            "nev": self.events,
            "checkpoint_cursor": self.events.len(),
            "state": {
                "public": self.public,
                "private": self.private,
            },
            "readout": self.readout(),
            "lane_status": {
                "phase": "fixture_parity",
                "supports_actions": true,
                "world_seed": self.world.seed,
                "max_steps": self.world.max_steps,
            },
        })
    }

    fn apply_move(&mut self, action: &str) -> Result<()> {
        let direction = dir_for_action(action).expect("direction action");
        self.world.player_direction = direction;
        let target = (
            self.world.player_pos.0 + direction.0,
            self.world.player_pos.1 + direction.1,
        );
        if let Some(entity_index) = self.entity_at(target, self.world.player_level) {
            let entity = self.world.entities[entity_index].clone();
            self.append_action(
                action,
                "blocked_by_mob",
                json!({"target": [target.0, target.1], "entity": entity.to_value()}),
            );
            return Ok(());
        }
        if !self.in_bounds(target) {
            self.append_action(
                action,
                "blocked",
                json!({"reason": "out_of_bounds", "target": [target.0, target.1]}),
            );
            return Ok(());
        }
        let block = self.block_at(target);
        if is_land_walkable(&block) {
            self.world.player_pos = target;
            self.append_action(
                action,
                "move",
                json!({"to": [target.0, target.1], "tile": self.tile_at(target), "block": block, "item": self.item_at(target)}),
            );
        } else {
            self.append_action(
                action,
                "blocked",
                json!({"reason": format!("blocked:{block}"), "target": [target.0, target.1], "tile": block}),
            );
        }
        Ok(())
    }

    fn apply_do(&mut self) -> Result<()> {
        let target = self.front_pos();
        if let Some(entity_index) = self.entity_at(target, self.world.player_level) {
            self.apply_melee(entity_index, "do")?;
            return Ok(());
        }
        if !self.in_bounds(target) {
            self.append_noop(
                "do",
                "out_of_bounds",
                json!({"target": [target.0, target.1]}),
            );
            return Ok(());
        }
        let tile = self.block_at(target);
        if let Some((resource, achievement, required_pickaxe)) = resource_tile(&tile) {
            if self.inventory_i64("pickaxe")? < required_pickaxe {
                self.append_noop(
                    "do",
                    &format!("needs_pickaxe:{}", tier_name(required_pickaxe)),
                    json!({"target": [target.0, target.1], "tile": tile}),
                );
                return Ok(());
            }
            self.add_inventory_i64(resource, 1)?;
            self.set_tile(target, resource_replacement(&tile, self.world.player_level));
            self.unlock(achievement)?;
            self.append_action(
                "do",
                "harvest",
                json!({"tile": tile, "resource": resource, "target": [target.0, target.1]}),
            );
            return Ok(());
        }
        if tile == "grass" {
            if self.rng.random() < 0.1 {
                self.add_inventory_i64("sapling", 1)?;
                self.unlock("collect_sapling")?;
                self.append_action(
                    "do",
                    "collect_sapling",
                    json!({"target": [target.0, target.1]}),
                );
            } else {
                self.append_noop("do", "no_sapling", json!({"target": [target.0, target.1]}));
            }
            return Ok(());
        }
        if tile == "water" || tile == "fountain" {
            let next = std::cmp::min(
                max_stat(&self.world, "drink")?,
                self.inventory_i64("drink")? + 1,
            );
            self.set_inventory_i64("drink", next)?;
            self.unlock("collect_drink")?;
            self.append_action(
                "do",
                "drink",
                json!({"tile": tile, "target": [target.0, target.1]}),
            );
            return Ok(());
        }
        if tile == "ripe_plant" {
            let next = std::cmp::min(
                max_stat(&self.world, "food")?,
                self.inventory_i64("food")? + 4,
            );
            self.set_inventory_i64("food", next)?;
            self.set_tile(target, "plant");
            self.unlock("eat_plant")?;
            self.append_action("do", "eat_plant", json!({"target": [target.0, target.1]}));
            return Ok(());
        }
        if tile == "chest" {
            self.loot_chest()?;
            self.set_tile(target, "path");
            let level =
                usize::try_from(self.world.player_level).context("player_level must fit usize")?;
            if level < self.world.chests_opened.len() {
                self.world.chests_opened[level] = true;
            }
            self.unlock("open_chest")?;
            self.append_action("do", "open_chest", json!({"target": [target.0, target.1]}));
            return Ok(());
        }
        if tile == "crafting_table" || tile == "furnace" {
            self.set_tile(target, "path");
            self.append_action(
                "do",
                "mine_block",
                json!({"tile": tile, "target": [target.0, target.1], "replacement": "path"}),
            );
            return Ok(());
        }
        if tile == "necromancer" {
            if self.world.player_level == self.world.levels - 1
                && self.active_mob_count("melee", self.world.player_level) == 0
                && self.active_mob_count("ranged", self.world.player_level) == 0
            {
                self.add_inventory_i64("boss_progress", 1)?;
                self.unlock("damage_necromancer")?;
                self.append_action(
                    "do",
                    "boss_damage",
                    json!({"target": [target.0, target.1], "boss_progress": self.inventory_i64("boss_progress")?}),
                );
                return Ok(());
            }
            self.append_action(
                "do",
                "boss_not_vulnerable",
                json!({"target": [target.0, target.1]}),
            );
            return Ok(());
        }
        self.append_noop(
            "do",
            &format!("nothing_to_do:{tile}"),
            json!({"target": [target.0, target.1], "tile": tile}),
        );
        Ok(())
    }

    fn apply_melee(&mut self, entity_index: usize, action: &str) -> Result<()> {
        // Entity transitions describe the combat result, not the submitted
        // action. Keep a primary action event so an action_applied-only replay
        // tape preserves this world-advancing step.
        let entity_before = self.world.entities[entity_index].clone();
        self.append_action(
            action,
            "melee",
            json!({"target": [entity_before.pos.0, entity_before.pos.1], "entity": entity_before.to_value()}),
        );
        if mob_class(&self.world.entities[entity_index].kind) == "passive" {
            let damage = self.player_damage()?;
            self.world.entities[entity_index].health -= damage;
            if self.world.entities[entity_index].health <= 0.0 {
                self.world.entities[entity_index].mask = false;
                let food = std::cmp::min(
                    max_stat(&self.world, "food")?,
                    self.inventory_i64("food")? + 6,
                );
                self.set_inventory_i64("food", food)?;
                let kind = self.world.entities[entity_index].kind.clone();
                if let Some(achievement) = passive_mob_achievement(&kind) {
                    self.unlock(achievement)?;
                }
                self.unlock("collect_food")?;
                let entity = self.world.entities[entity_index].clone();
                self.append_entity(action, &entity, "eat_passive");
            } else {
                let entity = self.world.entities[entity_index].clone();
                self.append_entity(action, &entity, "damage");
            }
            return Ok(());
        }
        let damage = self.player_damage()?;
        self.damage_entity(entity_index, damage, action)?;
        Ok(())
    }

    fn player_damage(&self) -> Result<f64> {
        let physical = match self.inventory_i64("sword")?.clamp(0, 4) {
            0 => 1.0,
            1 => 2.0,
            2 => 3.0,
            3 => 5.0,
            _ => 8.0,
        };
        Ok(physical * (1.0 + 0.25 * (self.inventory_i64("strength")? - 1) as f64))
    }

    fn loot_chest(&mut self) -> Result<()> {
        let level =
            usize::try_from(self.world.player_level).context("player_level must fit usize")?;
        let first_chest_on_level = self
            .world
            .chests_opened
            .get(level)
            .map(|opened| !*opened)
            .unwrap_or(true);

        let looting_wood = self.rng.random() < 0.6;
        let wood_amount = self.rng.randrange(1, 6);
        if looting_wood {
            self.add_inventory_i64("wood", wood_amount)?;
        }

        let looting_torch = self.rng.random() < 0.6;
        let torch_amount = self.rng.randrange(4, 8);
        if looting_torch {
            self.add_inventory_i64("torches", torch_amount)?;
        }

        let looting_ore = self.rng.random() < 0.6;
        let (ore, min_amount, max_amount) = self.weighted_chest_ore();
        let ore_amount = self.rng.randrange(min_amount, max_amount);
        if looting_ore {
            self.add_inventory_i64(ore, ore_amount)?;
        }

        let looting_potion = self.rng.random() < 0.5;
        let potion_color = POTION_COLORS[self.rng.randrange(0, 6) as usize];
        let potion_amount = self.rng.randrange(1, 3);
        if looting_potion {
            let potions = self
                .world
                .inventory
                .get_mut("potions")
                .and_then(Value::as_object_mut)
                .ok_or_else(|| anyhow!("inventory.potions must be object"))?;
            let count = potions
                .get(potion_color)
                .and_then(Value::as_i64)
                .unwrap_or(0);
            potions.insert(potion_color.to_string(), json!(count + potion_amount));
        }

        let looting_arrows = self.rng.random() < 0.25;
        let arrows_amount = self.rng.randrange(1, 5);
        if looting_arrows {
            self.add_inventory_i64("arrows", arrows_amount)?;
        }

        let looting_tool = self.rng.random() < 0.2;
        let tool_id = self.rng.randrange(0, 2);
        let pickaxe_level = self.weighted_tool_level();
        let sword_level = self.weighted_tool_level();
        if looting_tool && tool_id == 0 {
            self.set_inventory_i64("pickaxe", self.inventory_i64("pickaxe")?.max(pickaxe_level))?;
        }
        if looting_tool && tool_id == 1 {
            self.set_inventory_i64("sword", self.inventory_i64("sword")?.max(sword_level))?;
        }

        if self.world.player_level == 1 && first_chest_on_level {
            self.set_inventory_i64("bow", 1)?;
        }
        if (self.world.player_level == 3 || self.world.player_level == 4) && first_chest_on_level {
            self.add_inventory_i64("books", 1)?;
        }
        Ok(())
    }

    fn weighted_chest_ore(&mut self) -> (&'static str, i64, i64) {
        let roll = self.rng.random();
        if roll < 0.3 {
            ("coal", 1, 4)
        } else if roll < 0.6 {
            ("iron", 1, 3)
        } else if roll < 0.75 {
            ("diamond", 1, 2)
        } else if roll < 0.875 {
            ("sapphire", 1, 2)
        } else {
            ("ruby", 1, 2)
        }
    }

    fn weighted_tool_level(&mut self) -> i64 {
        let roll = self.rng.random();
        if roll < 0.4 {
            1
        } else if roll < 0.7 {
            2
        } else if roll < 0.9 {
            3
        } else {
            4
        }
    }

    fn apply_place(&mut self, action: &str) -> Result<()> {
        let target = self.front_pos();
        if !self.in_bounds(target) {
            self.append_noop(
                action,
                "out_of_bounds",
                json!({"target": [target.0, target.1]}),
            );
            return Ok(());
        }
        if let Some(entity_index) = self.entity_at(target, self.world.player_level) {
            let entity = self.world.entities[entity_index].clone();
            self.append_action(
                action,
                "blocked_by_mob",
                json!({"target": [target.0, target.1], "entity": entity.to_value()}),
            );
            return Ok(());
        }
        let target_tile = self.block_at(target);
        let target_item = self.item_at(target);
        let (costs, tile) = match action {
            "place_stone" => (vec![("stone", 1)], "stone"),
            "place_table" => (vec![("wood", 2)], "crafting_table"),
            "place_furnace" => (vec![("stone", 1)], "furnace"),
            "place_plant" => (vec![("sapling", 1)], "plant"),
            "place_torch" => (vec![("torches", 1)], "torch"),
            _ => {
                self.reject(action, "unknown_action")?;
                return Ok(());
            }
        };
        let mut valid_target = base_walkable(&target_tile) && target_item == "none";
        if action == "place_stone" {
            valid_target =
                target_tile == "water" || (base_walkable(&target_tile) && target_item == "none");
        } else if action == "place_plant" {
            valid_target = target_tile == "grass" && target_item == "none";
        } else if action == "place_torch" {
            valid_target = can_place_item_on(&target_tile) && target_item == "none";
        }
        if !valid_target {
            self.append_noop(
                action,
                &format!("target_not_placeable:{target_tile}"),
                json!({"target": [target.0, target.1], "tile": target_tile, "item": target_item}),
            );
            return Ok(());
        }
        if !self.pay(&costs)? {
            self.append_missing_resources(action, &costs)?;
            return Ok(());
        }
        if action == "place_torch" {
            self.set_item(target, "torch");
        } else {
            self.set_tile(target, tile);
        }
        self.unlock(action)?;
        self.append_action(
            action,
            "place",
            json!({"tile": tile, "target": [target.0, target.1]}),
        );
        Ok(())
    }

    fn apply_craft(&mut self, action: &str) -> Result<()> {
        if !self.near_tile(&["crafting_table"]) {
            self.append_noop(action, "needs_crafting_table", json!({}));
            return Ok(());
        }
        let Some(recipe) = recipe_for(action) else {
            self.reject(action, "unknown_action")?;
            return Ok(());
        };
        if action.starts_with("make_iron_") && !self.near_tile(&["furnace"]) {
            self.append_noop(action, "needs_furnace", json!({}));
            return Ok(());
        }
        if (recipe.target == "pickaxe" || recipe.target == "sword")
            && recipe.tier > 0
            && self.inventory_i64(recipe.target)? >= recipe.tier
        {
            self.append_noop(
                action,
                "already_have_tier",
                json!({"target": recipe.target, "tier": recipe.tier, "inventory_value": self.inventory_i64(recipe.target)?}),
            );
            return Ok(());
        }
        if (recipe.target == "arrows" || recipe.target == "torches")
            && self.inventory_i64(recipe.target)? >= 99
        {
            self.append_noop(
                action,
                "inventory_cap",
                json!({"target": recipe.target, "inventory_value": self.inventory_i64(recipe.target)?}),
            );
            return Ok(());
        }
        if !self.pay(recipe.costs)? {
            self.append_missing_resources(action, recipe.costs)?;
            return Ok(());
        }
        if recipe.target == "pickaxe" || recipe.target == "sword" {
            let current = self.inventory_i64(recipe.target)?;
            self.set_inventory_i64(recipe.target, current.max(recipe.tier))?;
        } else if recipe.target == "armour" {
            let mut armour = armour_vec(&self.world)?;
            for piece in &mut armour {
                if *piece < recipe.tier {
                    *piece = recipe.tier;
                    break;
                }
            }
            self.set_inventory_value("armour", json!(armour))?;
        } else {
            self.add_inventory_i64(recipe.target, recipe.amount)?;
        }
        self.unlock(action)?;
        self.append_action(
            action,
            "craft",
            json!({"target": recipe.target, "inventory_value": self.world.inventory.get(recipe.target).cloned().unwrap_or(Value::Null)}),
        );
        Ok(())
    }

    fn apply_recover(&mut self, action: &str) -> Result<()> {
        let before_energy = self.inventory_i64("energy")?;
        let energy_gain = if action == "sleep" { 3 } else { 1 };
        self.set_inventory_number(
            "energy",
            std::cmp::min(
                max_stat(&self.world, "energy")?,
                before_energy + energy_gain,
            ) as f64,
        )?;
        let next_health = std::cmp::min(
            max_stat(&self.world, "health")?,
            self.inventory_i64("health")? + 1,
        );
        if next_health != self.inventory_i64("health")? {
            self.set_inventory_number("health", next_health as f64)?;
        }
        let next_mana = std::cmp::min(
            max_stat(&self.world, "mana")?,
            self.inventory_i64("mana")? + 1,
        );
        if next_mana != self.inventory_i64("mana")? {
            self.set_inventory_number("mana", next_mana as f64)?;
        }
        if action == "sleep"
            && before_energy < max_stat(&self.world, "energy")?
            && self.inventory_i64("energy")? >= max_stat(&self.world, "energy")?
        {
            self.unlock("wake_up")?;
        }
        self.append_action(
            action,
            "recover",
            json!({"energy": self.world.inventory["energy"], "health": self.world.inventory["health"], "mana": self.world.inventory["mana"]}),
        );
        Ok(())
    }

    fn apply_read_book(&mut self) -> Result<()> {
        if self.inventory_i64("books")? <= 0 {
            self.append_noop("read_book", "missing_book", json!({}));
            return Ok(());
        }
        self.add_inventory_i64("books", -1)?;
        let mut learned_now = Vec::new();
        let mut spells = self
            .world
            .inventory
            .get("learned_spells")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if spells.is_empty() {
            spells.push(json!("fireball"));
            learned_now.push("fireball");
            self.unlock("learn_spell")?;
        }
        self.set_inventory_value("learned_spells", Value::Array(spells.clone()))?;
        self.append_action(
            "read_book",
            "learn_spell",
            json!({"learned_spells": spells, "learned_now": learned_now}),
        );
        Ok(())
    }

    fn apply_cast_spell(&mut self) -> Result<()> {
        let spells = self
            .world
            .inventory
            .get("learned_spells")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if spells.is_empty() || self.inventory_i64("mana")? < 2 {
            self.append_noop("cast_spell", "spell_not_ready:fireball", json!({}));
            return Ok(());
        }
        if self.spawn_player_projectile("fireball", "cast_spell")? {
            self.set_inventory_number("mana", self.inventory_f64("mana")? - 2.0)?;
            self.unlock("cast_spell")?;
            self.append_action(
                "cast_spell",
                "projectile_spawn",
                json!({"projectile": "fireball"}),
            );
        }
        Ok(())
    }

    fn apply_shoot_arrow(&mut self) -> Result<()> {
        if self.inventory_i64("bow")? <= 0 || self.inventory_i64("arrows")? <= 0 {
            self.append_action(
                "shoot_arrow",
                "noop",
                json!({"reason": "needs_bow_and_arrow"}),
            );
            return Ok(());
        }
        if self.spawn_player_projectile("arrow2", "shoot_arrow")? {
            self.add_inventory_i64("arrows", -1)?;
            self.unlock("fire_bow")?;
            self.append_action(
                "shoot_arrow",
                "projectile_spawn",
                json!({"projectile": "arrow2"}),
            );
        }
        Ok(())
    }

    fn apply_descend(&mut self) -> Result<()> {
        if self.item_at(self.world.player_pos) != "ladder_down" {
            self.append_noop(
                "descend",
                "not_on_ladder_down",
                json!({"pos": [self.world.player_pos.0, self.world.player_pos.1], "item": self.item_at(self.world.player_pos)}),
            );
            return Ok(());
        }
        let god_mode = self
            .resolved
            .rules
            .get("god_mode")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let level =
            usize::try_from(self.world.player_level).context("player_level must fit usize")?;
        if !god_mode && self.world.monsters_killed.get(level).copied().unwrap_or(0) < 8 {
            self.append_noop(
                "descend",
                "level_not_cleared",
                json!({"level": self.world.player_level, "monsters_killed": self.world.monsters_killed.get(level).copied().unwrap_or(0)}),
            );
            return Ok(());
        }
        if self.world.player_level + 1 >= self.world.levels {
            self.append_noop(
                "descend",
                "lowest_level",
                json!({"level": self.world.player_level}),
            );
            return Ok(());
        }
        self.world.player_level += 1;
        let next_level =
            usize::try_from(self.world.player_level).context("player_level must fit usize")?;
        self.world.player_pos = valid_ladder_pos(self.world.up_ladders.get(next_level).copied())
            .or_else(|| find_item(&self.world.item_maps[next_level], "ladder_up"))
            .unwrap_or((2, 2));
        if let Some(achievement) = level_achievement(self.world.player_level) {
            let had_achievement = self
                .world
                .achievements
                .get(achievement)
                .and_then(Value::as_i64)
                .unwrap_or(0)
                > 0;
            self.unlock(achievement)?;
            if !had_achievement {
                self.add_inventory_i64("xp", 1)?;
            }
        }
        self.append_floor_transition("descend");
        Ok(())
    }

    fn apply_ascend(&mut self) -> Result<()> {
        if self.item_at(self.world.player_pos) != "ladder_up" {
            self.append_noop(
                "ascend",
                "not_on_ladder_up",
                json!({"pos": [self.world.player_pos.0, self.world.player_pos.1], "item": self.item_at(self.world.player_pos)}),
            );
            return Ok(());
        }
        if self.world.player_level <= 0 {
            self.append_noop(
                "ascend",
                "top_level",
                json!({"level": self.world.player_level}),
            );
            return Ok(());
        }
        self.world.player_level -= 1;
        let next_level =
            usize::try_from(self.world.player_level).context("player_level must fit usize")?;
        self.world.player_pos = valid_ladder_pos(self.world.down_ladders.get(next_level).copied())
            .or_else(|| find_item(&self.world.item_maps[next_level], "ladder_down"))
            .unwrap_or((self.world.width - 3, self.world.height - 3));
        self.append_floor_transition("ascend");
        Ok(())
    }

    fn apply_drink_potion(&mut self, action: &str) -> Result<()> {
        let color = action
            .strip_prefix("drink_potion_")
            .ok_or_else(|| anyhow!("invalid potion action"))?;
        let potions = self
            .world
            .inventory
            .get_mut("potions")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| anyhow!("inventory.potions must be object"))?;
        let count = potions.get(color).and_then(Value::as_i64).unwrap_or(0);
        if count <= 0 {
            self.append_noop(action, "missing_potion", json!({"color": color}));
            return Ok(());
        }
        potions.insert(color.to_string(), json!(count - 1));
        let effect = self.world.potion_mapping[potion_index(color)?];
        match effect {
            0 => self.set_inventory_number(
                "health",
                (max_stat(&self.world, "health")? as f64).min(self.inventory_f64("health")? + 8.0),
            )?,
            1 => self
                .set_inventory_number("health", 0.0_f64.max(self.inventory_f64("health")? - 3.0))?,
            2 => self.set_inventory_number(
                "mana",
                (max_stat(&self.world, "mana")? as f64).min(self.inventory_f64("mana")? + 8.0),
            )?,
            3 => {
                self.set_inventory_number("mana", 0.0_f64.max(self.inventory_f64("mana")? - 3.0))?
            }
            4 => self.set_inventory_number(
                "energy",
                (max_stat(&self.world, "energy")? as f64).min(self.inventory_f64("energy")? + 8.0),
            )?,
            5 => self
                .set_inventory_number("energy", 0.0_f64.max(self.inventory_f64("energy")? - 3.0))?,
            _ => {}
        }
        self.unlock("drink_potion")?;
        self.append_action(
            action,
            "drink_potion",
            json!({"color": color, "effect_index": effect, "potion_mapping": self.world.potion_mapping}),
        );
        Ok(())
    }

    fn apply_enchant(&mut self, action: &str) -> Result<()> {
        let item = action.strip_prefix("enchant_").unwrap_or(action);
        let target_tile = if self.in_bounds(self.front_pos()) {
            self.tile_at(self.front_pos())
        } else {
            "out_of_bounds".to_string()
        };
        if target_tile != "enchantment_table_fire" && target_tile != "enchantment_table_ice" {
            self.append_noop(
                action,
                "needs_enchantment_table",
                json!({"front_tile": target_tile}),
            );
            return Ok(());
        }
        if self.inventory_i64("mana")? < 9 {
            self.append_noop(
                action,
                "needs_mana",
                json!({"mana": self.world.inventory["mana"]}),
            );
            return Ok(());
        }
        let enchantment = if target_tile == "enchantment_table_fire" {
            "fire"
        } else {
            "ice"
        };
        let gem = if enchantment == "fire" {
            "ruby"
        } else {
            "sapphire"
        };
        if self.inventory_i64(gem)? <= 0 {
            self.append_noop(action, &format!("missing_{gem}"), json!({"gem": gem}));
            return Ok(());
        }
        if item == "sword" && self.inventory_i64("sword")? <= 0 {
            self.append_noop(action, "missing_sword", json!({}));
            return Ok(());
        }
        if item == "bow" && self.inventory_i64("bow")? <= 0 {
            self.append_noop(action, "missing_bow", json!({}));
            return Ok(());
        }
        if item == "armour" && armour_vec(&self.world)?.iter().sum::<i64>() <= 0 {
            self.append_noop(
                action,
                "missing_armour",
                json!({"armour": armour_vec(&self.world)?}),
            );
            return Ok(());
        }
        self.add_inventory_i64(gem, -1)?;
        self.set_inventory_number("mana", self.inventory_f64("mana")? - 9.0)?;
        if item == "armour" {
            let mut enchants = self
                .world
                .inventory
                .get("armour_enchantments")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_else(|| {
                    vec![json!("none"), json!("none"), json!("none"), json!("none")]
                });
            let mut targets: Vec<usize> = enchants
                .iter()
                .enumerate()
                .filter_map(|(idx, value)| (value.as_str() == Some("none")).then_some(idx))
                .collect();
            if targets.is_empty() {
                targets = enchants
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, value)| {
                        let current = value.as_str().unwrap_or("none");
                        (current != "none" && current != enchantment).then_some(idx)
                    })
                    .collect();
            }
            let target_index = targets[self.rng.randrange(0, targets.len() as i64) as usize];
            enchants[target_index] = json!(enchantment);
            self.set_inventory_value("armour_enchantments", Value::Array(enchants))?;
        } else {
            self.set_inventory_value(&format!("{item}_enchantment"), json!(enchantment))?;
        }
        if action == "enchant_sword" || action == "enchant_armour" {
            self.unlock(action)?;
        }
        self.append_action(
            action,
            "enchant",
            json!({"item": item, "gem": gem, "enchantment": enchantment}),
        );
        Ok(())
    }

    fn apply_level_up(&mut self, action: &str) -> Result<()> {
        let attr = action
            .strip_prefix("level_up_")
            .ok_or_else(|| anyhow!("invalid level_up action"))?;
        if self.inventory_i64("xp")? <= 0 {
            self.append_noop(action, "missing_xp", json!({"attribute": attr}));
            return Ok(());
        }
        if self.inventory_i64(attr)? >= 5 {
            self.append_noop(
                action,
                "attribute_max",
                json!({"attribute": attr, "value": self.world.inventory[attr]}),
            );
            return Ok(());
        }
        self.add_inventory_i64("xp", -1)?;
        self.add_inventory_i64(attr, 1)?;
        self.append_action(
            action,
            "level_up",
            json!({"attribute": attr, "value": self.world.inventory[attr]}),
        );
        Ok(())
    }

    fn front_pos(&self) -> (i64, i64) {
        (
            self.world.player_pos.0 + self.world.player_direction.0,
            self.world.player_pos.1 + self.world.player_direction.1,
        )
    }

    fn observation(&self) -> Value {
        let front = self.front_pos();
        let achievements = self
            .world
            .achievements
            .as_object()
            .map(|object| {
                let mut names: Vec<String> = object
                    .iter()
                    .filter_map(|(name, value)| {
                        (value.as_i64().unwrap_or(0) > 0).then_some(name.clone())
                    })
                    .collect();
                names.sort();
                names
            })
            .unwrap_or_default();
        json!({
            "player": {
                "pos": [self.world.player_pos.0, self.world.player_pos.1],
                "level": self.world.player_level,
                "direction": [self.world.player_direction.0, self.world.player_direction.1],
                "front_tile": if self.in_bounds(front) { self.tile_at(front) } else { "out_of_bounds".to_string() },
                "front_block": if self.in_bounds(front) { self.block_at(front) } else { "out_of_bounds".to_string() },
                "front_item": if self.in_bounds(front) { self.item_at(front) } else { "none".to_string() },
            },
            "local_map": self.local_map(None),
            "inventory": self.world.inventory,
            "intrinsics": {
                "is_sleeping": self.world.is_sleeping,
                "is_resting": self.world.is_resting,
                "recover": self.world.player_recover,
                "hunger": self.world.player_hunger,
                "thirst": self.world.player_thirst,
                "fatigue": self.world.player_fatigue,
                "recover_mana": self.world.player_recover_mana,
            },
            "potion_mapping": self.world.potion_mapping,
            "floor_state": {
                "monsters_killed": self.world.monsters_killed,
                "chests_opened": self.world.chests_opened,
                "down_ladders": self.world.down_ladders.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>(),
                "up_ladders": self.world.up_ladders.iter().map(|(x, y)| json!([x, y])).collect::<Vec<_>>(),
                "growing_plants": [],
            },
            "mob_state": {
                "entities": self.world.entities.iter().filter(|entity| entity.mask).map(Entity::to_value).collect::<Vec<_>>(),
            },
            "projectile_state": {
                "player": self.world.player_projectiles.iter().filter(|projectile| projectile.mask).map(Projectile::to_value).collect::<Vec<_>>(),
                "mob": self.world.mob_projectiles.iter().filter(|projectile| projectile.mask).map(Projectile::to_value).collect::<Vec<_>>(),
            },
            "achievements": achievements,
            "nearby_entities": self.entities_near(5),
        })
    }

    fn local_map(&self, radius: Option<i64>) -> Vec<String> {
        let r = radius.unwrap_or_else(|| {
            self.resolved
                .world
                .get("view_radius")
                .and_then(Value::as_i64)
                .unwrap_or(4)
        });
        let mut rows = Vec::new();
        for y in (self.world.player_pos.1 - r)..=(self.world.player_pos.1 + r) {
            let mut row = String::new();
            for x in (self.world.player_pos.0 - r)..=(self.world.player_pos.0 + r) {
                let pos = (x, y);
                if pos == self.world.player_pos {
                    row.push('P');
                } else if let Some(index) = self.entity_at(pos, self.world.player_level) {
                    row.push(entity_char(&self.world.entities[index].kind));
                } else if let Some(projectile) = self.projectile_at(pos, self.world.player_level) {
                    row.push(projectile_char(&projectile.kind, &projectile.owner));
                } else {
                    row.push(tile_char(&self.tile_at(pos)));
                }
            }
            rows.push(row);
        }
        rows
    }

    fn projectile_at(&self, pos: (i64, i64), level: i64) -> Option<&Projectile> {
        self.world
            .player_projectiles
            .iter()
            .chain(self.world.mob_projectiles.iter())
            .find(|projectile| {
                projectile.mask && projectile.level == level && projectile.pos == pos
            })
    }

    fn entities_near(&self, radius: i64) -> Vec<Value> {
        self.world
            .entities
            .iter()
            .filter(|entity| {
                entity.mask
                    && entity.level == self.world.player_level
                    && manhattan(entity.pos, self.world.player_pos) <= radius
            })
            .map(Entity::to_value)
            .collect()
    }

    fn valid_actions(&self) -> Vec<&'static str> {
        if self.is_done() {
            Vec::new()
        } else {
            ACTION_NAMES.to_vec()
        }
    }

    fn in_bounds(&self, pos: (i64, i64)) -> bool {
        pos.0 >= 0 && pos.0 < self.world.width && pos.1 >= 0 && pos.1 < self.world.height
    }

    fn block_at(&self, pos: (i64, i64)) -> String {
        if !self.in_bounds(pos) {
            return "out_of_bounds".to_string();
        }
        self.world.maps[self.world.player_level as usize][pos.1 as usize][pos.0 as usize].clone()
    }

    fn item_at(&self, pos: (i64, i64)) -> String {
        if !self.in_bounds(pos) {
            return "none".to_string();
        }
        self.world.item_maps[self.world.player_level as usize][pos.1 as usize][pos.0 as usize]
            .clone()
    }

    fn tile_at(&self, pos: (i64, i64)) -> String {
        let item = self.item_at(pos);
        if item != "none" {
            item
        } else {
            self.block_at(pos)
        }
    }

    fn set_tile(&mut self, pos: (i64, i64), tile: &str) {
        self.world.maps[self.world.player_level as usize][pos.1 as usize][pos.0 as usize] =
            tile.to_string();
    }

    fn set_item(&mut self, pos: (i64, i64), item: &str) {
        self.world.item_maps[self.world.player_level as usize][pos.1 as usize][pos.0 as usize] =
            item.to_string();
    }

    fn near_tile(&self, tiles: &[&str]) -> bool {
        [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
            .iter()
            .any(|(dx, dy)| {
                let pos = (self.world.player_pos.0 + dx, self.world.player_pos.1 + dy);
                self.in_bounds(pos) && tiles.contains(&self.block_at(pos).as_str())
            })
    }

    fn append_action(&mut self, action: &str, transition: &str, payload: Value) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "action_applied".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some(transition.to_string()),
            severity: "info".to_string(),
            message: format!("ActionApplied({action},step={})", self.world.timestep),
            payload,
        });
    }

    fn append_noop(&mut self, action: &str, reason: &str, payload: Value) {
        let mut object = Map::new();
        object.insert("reason".to_string(), Value::String(reason.to_string()));
        if let Value::Object(extra) = payload {
            for (key, value) in extra {
                object.insert(key, value);
            }
        }
        self.append_action(action, "noop", Value::Object(object));
    }

    fn append_achievement(&mut self, action: &str, achievement: &str) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "achievement_unlocked".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some(achievement.to_string()),
            severity: "info".to_string(),
            message: format!("AchievementUnlocked({achievement})"),
            payload: json!({"achievement": achievement}),
        });
    }

    fn append_entity(&mut self, action: &str, entity: &Entity, transition: &str) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "entity_transition".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some(transition.to_string()),
            severity: "info".to_string(),
            message: format!("EntityTransition({transition},{})", entity.kind),
            payload: json!({"entity": entity.to_value()}),
        });
    }

    fn append_projectile(
        &mut self,
        action: &str,
        projectile: &Projectile,
        transition: &str,
        extra: Value,
    ) {
        let mut payload = match extra {
            Value::Object(object) => object,
            _ => Map::new(),
        };
        payload.insert("projectile".to_string(), projectile.to_value());
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "projectile_transition".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some(transition.to_string()),
            severity: "info".to_string(),
            message: format!("ProjectileTransition({transition},{})", projectile.kind),
            payload: Value::Object(payload),
        });
    }

    fn append_floor_transition(&mut self, action: &str) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "floor_transition".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some(action.to_string()),
            severity: "info".to_string(),
            message: format!("FloorTransition({action},{})", self.world.player_level),
            payload: json!({"level": self.world.player_level}),
        });
    }

    fn append_terminal_event(&mut self, kind: &str, message: &str, reason: &str) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: kind.to_string(),
            action: None,
            transition: Some(reason.to_string()),
            severity: "info".to_string(),
            message: message.to_string(),
            payload: json!({"reason": reason}),
        });
    }

    fn append_combat(&mut self, action: &str, entity: &Entity, damage: f64) {
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "combat".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some("mob_attack".to_string()),
            severity: "info".to_string(),
            message: format!("MobAttack({},{:.2})", entity.kind, damage),
            payload: json!({"entity": entity.to_value(), "damage": damage}),
        });
    }

    fn append_inventory_deltas(
        &mut self,
        action: &str,
        before: &Value,
        after: &Value,
    ) -> Result<()> {
        let before_obj = as_object(before, "before_inventory")?;
        let after_obj = as_object(after, "after_inventory")?;
        let mut keys: Vec<String> = before_obj.keys().chain(after_obj.keys()).cloned().collect();
        keys.sort();
        keys.dedup();
        for key in keys {
            let before_value = before_obj.get(&key).cloned().unwrap_or(Value::Null);
            let after_value = after_obj.get(&key).cloned().unwrap_or(Value::Null);
            if before_value == after_value {
                continue;
            }
            if key == "potions" || key == "learned_spells" {
                self.events.push(EventRecord {
                    step_index: self.world.timestep,
                    tick: self.world.timestep,
                    episode_id: self.resolved.episode_id.clone(),
                    kind: "resource_delta".to_string(),
                    action: Some(Value::String(action.to_string())),
                    transition: Some(key.clone()),
                    severity: "info".to_string(),
                    message: format!("ResourceDelta({key})"),
                    payload: json!({"before": before_value, "after": after_value}),
                });
                continue;
            }
            let delta = match (before_value.as_i64(), after_value.as_i64()) {
                (Some(before), Some(after)) => json!(after - before),
                _ => match (before_value.as_f64(), after_value.as_f64()) {
                    (Some(before), Some(after)) => Value::Number(
                        Number::from_f64(after - before)
                            .ok_or_else(|| anyhow!("non-finite delta"))?,
                    ),
                    _ => Value::Null,
                },
            };
            self.events.push(EventRecord {
                step_index: self.world.timestep,
                tick: self.world.timestep,
                episode_id: self.resolved.episode_id.clone(),
                kind: "resource_delta".to_string(),
                action: Some(Value::String(action.to_string())),
                transition: Some(key.clone()),
                severity: "info".to_string(),
                message: format!(
                    "ResourceDelta({},{})",
                    key,
                    if delta.is_null() {
                        "None".to_string()
                    } else {
                        delta.to_string()
                    }
                ),
                payload: json!({"resource": key, "before": before_value, "after": after_value, "delta": delta}),
            });
        }
        Ok(())
    }

    fn calculate_inventory_achievements(&mut self) -> Result<()> {
        for (resource, achievement) in [
            ("wood", "collect_wood"),
            ("stone", "collect_stone"),
            ("coal", "collect_coal"),
            ("iron", "collect_iron"),
            ("diamond", "collect_diamond"),
            ("ruby", "collect_ruby"),
            ("sapphire", "collect_sapphire"),
            ("sapling", "collect_sapling"),
        ] {
            if self.inventory_i64(resource)? > 0 {
                self.unlock(achievement)?;
            }
        }
        if self.inventory_i64("bow")? > 0 {
            self.unlock("find_bow")?;
        }
        if self.inventory_i64("arrows")? > 0 {
            self.unlock("make_arrow")?;
        }
        if self.inventory_i64("torches")? > 0 {
            self.unlock("make_torch")?;
        }
        for (threshold, achievement) in [
            (1, "make_wood_pickaxe"),
            (2, "make_stone_pickaxe"),
            (3, "make_iron_pickaxe"),
            (4, "make_diamond_pickaxe"),
        ] {
            if self.inventory_i64("pickaxe")? >= threshold {
                self.unlock(achievement)?;
            }
        }
        for (threshold, achievement) in [
            (1, "make_wood_sword"),
            (2, "make_stone_sword"),
            (3, "make_iron_sword"),
            (4, "make_diamond_sword"),
        ] {
            if self.inventory_i64("sword")? >= threshold {
                self.unlock(achievement)?;
            }
        }
        Ok(())
    }

    fn update_mobs(&mut self, action: &str) -> Result<()> {
        let len = self.world.entities.len();
        for index in 0..len {
            if self.entity_active_class(index, "melee") {
                self.update_melee_mob(index, action)?;
            }
        }
        for index in 0..len {
            if self.entity_active_class(index, "passive") {
                self.update_passive_mob(index, action);
            }
        }
        for index in 0..len {
            if self.entity_active_class(index, "ranged") {
                self.update_ranged_mob(index, action);
            }
        }
        Ok(())
    }

    fn spawn_player_projectile(&mut self, kind: &str, action: &str) -> Result<bool> {
        let capacity = self
            .resolved
            .world
            .get("max_player_projectiles")
            .and_then(Value::as_i64)
            .unwrap_or(3);
        let active_count = self
            .world
            .player_projectiles
            .iter()
            .filter(|projectile| projectile.mask && projectile.level == self.world.player_level)
            .count() as i64;
        if active_count >= capacity {
            self.append_action(
                action,
                "noop",
                json!({"reason": "projectile_capacity_full", "capacity": capacity}),
            );
            return Ok(false);
        }
        let projectile = Projectile {
            id: format!(
                "player_projectile_{}_{}_{}",
                self.world.player_level,
                self.world.timestep,
                self.world.player_projectiles.len()
            ),
            kind: kind.to_string(),
            owner: "player".to_string(),
            level: self.world.player_level,
            pos: self.world.player_pos,
            direction: self.world.player_direction,
            mask: true,
        };
        if let Some(index) = self
            .world
            .player_projectiles
            .iter()
            .position(|existing| !existing.mask && existing.level == self.world.player_level)
        {
            self.world.player_projectiles[index] = projectile.clone();
        } else {
            self.world.player_projectiles.push(projectile.clone());
        }
        self.append_projectile(
            action,
            &projectile,
            "spawn",
            json!({"pos": [projectile.pos.0, projectile.pos.1], "direction": [projectile.direction.0, projectile.direction.1]}),
        );
        Ok(true)
    }

    fn update_player_projectiles(&mut self, action: &str) -> Result<()> {
        let len = self.world.player_projectiles.len();
        for index in 0..len {
            if !self.world.player_projectiles[index].mask
                || self.world.player_projectiles[index].level != self.world.player_level
            {
                continue;
            }
            let old_pos = self.world.player_projectiles[index].pos;
            let direction = self.world.player_projectiles[index].direction;
            let proposed = (old_pos.0 + direction.0, old_pos.1 + direction.1);
            if let Some(entity_index) = self.entity_at(old_pos, self.world.player_level) {
                let damage = projectile_damage(&self.world.player_projectiles[index].kind);
                let entity = self.damage_entity(entity_index, damage, action)?;
                self.world.player_projectiles[index].pos = proposed;
                self.world.player_projectiles[index].mask = false;
                let projectile = self.world.player_projectiles[index].clone();
                self.append_projectile(
                    action,
                    &projectile,
                    "hit_mob",
                    json!({"from": [old_pos.0, old_pos.1], "hit_pos": [old_pos.0, old_pos.1], "to": [proposed.0, proposed.1], "entity": entity.to_value()}),
                );
                continue;
            }
            let proposed_entity = if self.in_bounds(proposed) {
                self.entity_at(proposed, self.world.player_level)
            } else {
                None
            };
            if let Some(entity_index) = proposed_entity {
                let damage = projectile_damage(&self.world.player_projectiles[index].kind);
                let entity = self.damage_entity(entity_index, damage, action)?;
                self.world.player_projectiles[index].pos = proposed;
                self.world.player_projectiles[index].mask = false;
                let projectile = self.world.player_projectiles[index].clone();
                self.append_projectile(
                    action,
                    &projectile,
                    "hit_mob",
                    json!({"from": [old_pos.0, old_pos.1], "hit_pos": [proposed.0, proposed.1], "to": [proposed.0, proposed.1], "entity": entity.to_value()}),
                );
                continue;
            }
            if !self.in_bounds(proposed) {
                self.world.player_projectiles[index].pos = proposed;
                self.world.player_projectiles[index].mask = false;
                let projectile = self.world.player_projectiles[index].clone();
                self.append_projectile(
                    action,
                    &projectile,
                    "despawn",
                    json!({"from": [old_pos.0, old_pos.1], "to": [proposed.0, proposed.1], "reason": "out_of_bounds"}),
                );
                continue;
            }
            let block = self.block_at(proposed);
            if static_solid(&block) {
                self.world.player_projectiles[index].pos = proposed;
                self.world.player_projectiles[index].mask = false;
                let projectile = self.world.player_projectiles[index].clone();
                self.append_projectile(
                    action,
                    &projectile,
                    "despawn",
                    json!({"from": [old_pos.0, old_pos.1], "to": [proposed.0, proposed.1], "reason": format!("blocked:{block}")}),
                );
                continue;
            }
            self.world.player_projectiles[index].pos = proposed;
            let projectile = self.world.player_projectiles[index].clone();
            self.append_projectile(
                action,
                &projectile,
                "move",
                json!({"from": [old_pos.0, old_pos.1], "to": [proposed.0, proposed.1]}),
            );
        }
        Ok(())
    }

    fn damage_entity(&mut self, index: usize, damage: f64, action: &str) -> Result<Entity> {
        self.world.entities[index].health -= damage;
        if self.world.entities[index].health <= 0.0 {
            self.world.entities[index].mask = false;
            let level =
                usize::try_from(self.world.player_level).context("player_level must fit usize")?;
            if mob_class(&self.world.entities[index].kind) != "passive"
                && level < self.world.monsters_killed.len()
            {
                self.world.monsters_killed[level] += 1;
                if let Some(achievement) = mob_achievement(&self.world.entities[index].kind) {
                    self.unlock(achievement)?;
                }
            }
            let entity = self.world.entities[index].clone();
            self.append_entity(action, &entity, "defeat");
            Ok(entity)
        } else {
            let entity = self.world.entities[index].clone();
            self.append_entity(action, &entity, "damage");
            Ok(entity)
        }
    }

    fn update_melee_mob(&mut self, index: usize, action: &str) -> Result<()> {
        let old_pos = self.world.entities[index].pos;
        let old_distance = manhattan(old_pos, self.world.player_pos);
        let random_direction =
            self.random_in_bounds_direction(old_pos, &[(0, -1), (0, 1), (-1, 0), (1, 0)]);
        let random_position = add_pos(old_pos, random_direction);
        let player_direction = self.mob_player_direction(old_pos);
        let player_position = add_pos(old_pos, player_direction);
        let mut proposed = if old_distance < 10 && self.rng.random() < 0.75 {
            player_position
        } else {
            random_position
        };
        let attacking = old_distance == 1
            && self.world.entities[index].attack_cooldown <= 0
            && self.world.entities[index].mask;
        if attacking {
            proposed = old_pos;
            let kind = self.world.entities[index].kind.clone();
            let damage = self.damage_player(melee_damage_vector(&kind))?;
            self.world.entities[index].attack_cooldown = 5;
            let entity = self.world.entities[index].clone();
            self.append_entity(action, &entity, "attack_player");
            self.append_combat(action, &entity, damage);
        } else {
            self.world.entities[index].attack_cooldown -= 1;
        }
        if !attacking && self.valid_mob_position(index, proposed, "melee") {
            self.world.entities[index].pos = proposed;
        }
        if old_distance >= self.mob_despawn_distance() {
            self.world.entities[index].mask = false;
            let entity = self.world.entities[index].clone();
            self.append_entity(action, &entity, "despawn");
        }
        Ok(())
    }

    fn update_passive_mob(&mut self, index: usize, action: &str) {
        let old_pos = self.world.entities[index].pos;
        let old_distance = manhattan(old_pos, self.world.player_pos);
        let direction = self.random_in_bounds_direction(
            old_pos,
            &[
                (0, -1),
                (0, 1),
                (-1, 0),
                (1, 0),
                (0, 0),
                (0, 0),
                (0, 0),
                (0, 0),
            ],
        );
        let proposed = add_pos(old_pos, direction);
        if self.valid_mob_position(index, proposed, "passive") {
            self.world.entities[index].pos = proposed;
        }
        if old_distance >= self.mob_despawn_distance() {
            self.world.entities[index].mask = false;
            let entity = self.world.entities[index].clone();
            self.append_entity(action, &entity, "despawn");
        }
    }

    fn update_ranged_mob(&mut self, index: usize, _action: &str) {
        let old_pos = self.world.entities[index].pos;
        let random_direction =
            self.random_in_bounds_direction(old_pos, &[(0, -1), (0, 1), (-1, 0), (1, 0)]);
        let random_position = add_pos(old_pos, random_direction);
        let player_direction = self.mob_player_direction(old_pos);
        let selected_axis_distance = if player_direction.0 != 0 {
            (self.world.player_pos.0 - old_pos.0).abs()
        } else {
            (self.world.player_pos.1 - old_pos.1).abs()
        };
        let towards_player = add_pos(old_pos, player_direction);
        let away_from_player = (
            old_pos.0 - player_direction.0,
            old_pos.1 - player_direction.1,
        );
        let far_from_player = selected_axis_distance >= 6;
        let too_close_to_player = selected_axis_distance <= 3;
        let mut proposed = if far_from_player {
            towards_player
        } else {
            random_position
        };
        if too_close_to_player {
            proposed = away_from_player;
        }
        if self.rng.random() <= 0.85 {
            proposed = random_position;
        }
        let attacking = !far_from_player
            && self.world.entities[index].attack_cooldown <= 0
            && self.world.entities[index].mask;
        if attacking {
            self.world.entities[index].attack_cooldown = 4;
        } else {
            self.world.entities[index].attack_cooldown -= 1;
        }
        if !attacking && self.valid_mob_position(index, proposed, "ranged") {
            self.world.entities[index].pos = proposed;
        }
        if manhattan(old_pos, self.world.player_pos) >= self.mob_despawn_distance() {
            self.world.entities[index].mask = false;
            let entity = self.world.entities[index].clone();
            self.append_entity(_action, &entity, "despawn");
        }
    }

    fn spawn_mobs(&mut self, action: &str) -> Result<()> {
        self.maybe_spawn_mob("passive", action)?;
        self.maybe_spawn_mob("melee", action)?;
        self.maybe_spawn_mob("ranged", action)?;
        Ok(())
    }

    fn maybe_spawn_mob(&mut self, mob_class_name: &str, action: &str) -> Result<()> {
        let capacity = self.mob_capacity(mob_class_name);
        if self.active_mob_count(mob_class_name, self.world.player_level) >= capacity {
            return Ok(());
        }
        let class_index = mob_class_index(mob_class_name);
        let mut spawn_chance = floor_spawn_chance(self.world.player_level, class_index);
        if mob_class_name == "melee" {
            spawn_chance += floor_spawn_chance(self.world.player_level, 3)
                * (1.0 - self.world.light_level).powi(2);
        }
        if mob_class_name == "melee" || mob_class_name == "ranged" {
            spawn_chance *= if self.world.monsters_killed[self.world.player_level as usize] < 8 {
                3.0
            } else {
                1.0
            };
        }
        if self.world.player_level == self.world.levels - 1 {
            if mob_class_name == "passive" {
                return Ok(());
            }
            if mob_class_name == "melee" || mob_class_name == "ranged" {
                spawn_chance = 0.0;
            }
        }
        if self.rng.random() >= spawn_chance {
            return Ok(());
        }
        let kind = spawn_mob_kind(self.world.player_level, mob_class_name);
        let candidates = self.spawn_candidates(mob_class_name, kind);
        if candidates.is_empty() {
            return Ok(());
        }
        let pos = candidates[self.rng.randrange(0, candidates.len() as i64) as usize];
        let entity = Entity {
            id: format!(
                "{}_{}_{}",
                kind,
                self.world.timestep,
                self.world.entities.len()
            ),
            kind: kind.to_string(),
            pos,
            health: mob_health(kind) as f64,
            level: self.world.player_level,
            mob_class: mob_class_name.to_string(),
            attack_cooldown: 0,
            mask: true,
        };
        self.world.entities.push(entity.clone());
        self.append_entity(action, &entity, "spawn");
        Ok(())
    }

    fn spawn_candidates(&self, mob_class_name: &str, kind: &str) -> Vec<(i64, i64)> {
        let mut candidates = Vec::new();
        for y in 0..self.world.height {
            for x in 0..self.world.width {
                let pos = (x, y);
                let block = self.block_at(pos);
                let distance = euclidean(pos, self.world.player_pos);
                if self.entity_at(pos, self.world.player_level).is_some()
                    || pos == self.world.player_pos
                {
                    continue;
                }
                if mob_class_name == "passive" {
                    if !matches!(
                        block.as_str(),
                        "grass" | "path" | "fire_grass" | "ice_grass"
                    ) {
                        continue;
                    }
                    if distance <= 3.0 || distance >= self.mob_despawn_distance() as f64 {
                        continue;
                    }
                } else {
                    if kind == "deep_thing" {
                        if block != "water" {
                            continue;
                        }
                    } else if !matches!(
                        block.as_str(),
                        "grass" | "path" | "fire_grass" | "ice_grass"
                    ) {
                        continue;
                    }
                    if distance <= 9.0 || distance >= self.mob_despawn_distance() as f64 {
                        continue;
                    }
                }
                candidates.push(pos);
            }
        }
        candidates
    }

    fn entity_active_class(&self, index: usize, class_name: &str) -> bool {
        let entity = &self.world.entities[index];
        entity.mask && entity.level == self.world.player_level && entity.mob_class == class_name
    }

    fn entity_at(&self, pos: (i64, i64), level: i64) -> Option<usize> {
        self.world
            .entities
            .iter()
            .position(|entity| entity.mask && entity.level == level && entity.pos == pos)
    }

    fn valid_mob_position(&self, index: usize, pos: (i64, i64), class_name: &str) -> bool {
        if !self.in_bounds(pos) || pos == self.world.player_pos {
            return false;
        }
        if self
            .entity_at(pos, self.world.entities[index].level)
            .is_some()
        {
            return false;
        }
        let block = self.block_at(pos);
        mob_can_occupy_block(&self.world.entities[index].kind, class_name, &block)
    }

    fn mob_player_direction(&mut self, pos: (i64, i64)) -> (i64, i64) {
        let dx = self.world.player_pos.0 - pos.0;
        let dy = self.world.player_pos.1 - pos.1;
        let abs_dx = dx.abs();
        let abs_dy = dy.abs();
        if abs_dx == 0 && abs_dy == 0 {
            return (0, 0);
        }
        let axis = if abs_dx == abs_dy {
            self.rng.randrange(0, 2)
        } else if abs_dx > abs_dy {
            0
        } else {
            1
        };
        if axis == 0 {
            (sign(dx), 0)
        } else {
            (0, sign(dy))
        }
    }

    fn random_in_bounds_direction(
        &mut self,
        pos: (i64, i64),
        directions: &[(i64, i64)],
    ) -> (i64, i64) {
        let candidates: Vec<(i64, i64)> = directions
            .iter()
            .copied()
            .filter(|direction| self.in_bounds(add_pos(pos, *direction)))
            .collect();
        if candidates.is_empty() {
            (0, 0)
        } else {
            candidates[self.rng.randrange(0, candidates.len() as i64) as usize]
        }
    }

    fn damage_player(&mut self, damage_vector: [f64; 3]) -> Result<f64> {
        let armour = armour_vec(&self.world)?;
        let defense = armour.iter().map(|piece| *piece as f64 * 0.1).sum::<f64>();
        let damage = (1.0 - defense) * damage_vector[0] + damage_vector[1] + damage_vector[2];
        let health = self.inventory_f64("health")? - damage;
        self.set_inventory_number("health", health)?;
        Ok(damage)
    }

    fn apply_health_reward(&mut self, before_health: f64) -> Result<()> {
        if before_health <= 0.0 {
            return Ok(());
        }
        let delta = self.inventory_f64("health")? - before_health;
        if delta != 0.0 {
            self.add_private_f64("reward_last", delta * 0.1)?;
            self.add_private_f64("total_reward", delta * 0.1)?;
        }
        Ok(())
    }

    /// Public JAX Craftax intrinsic equations, preserving their update order.
    fn update_intrinsics(&mut self, action: &str) -> Result<()> {
        let max_health = max_stat(&self.world, "health")? as f64;
        let max_energy = max_stat(&self.world, "energy")? as f64;
        let max_mana = max_stat(&self.world, "mana")? as f64;

        if action == "sleep" && self.inventory_f64("energy")? < max_energy {
            self.world.is_sleeping = true;
        }
        if self.world.is_sleeping && self.inventory_f64("energy")? >= max_energy {
            self.world.is_sleeping = false;
            self.unlock("wake_up")?;
        }
        if action == "rest" && self.inventory_f64("health")? < max_health {
            self.world.is_resting = true;
        }
        if self.world.is_resting
            && (self.inventory_f64("health")? >= max_health
                || self.inventory_f64("food")? <= 0.0
                || self.inventory_f64("drink")? <= 0.0)
        {
            self.world.is_resting = false;
        }

        let not_boss = self.world.player_level != self.world.levels - 1;
        let decay_coeff = 1.0 - 0.125 * (self.inventory_i64("dexterity")? - 1) as f64;
        let sleep_coeff = if self.world.is_sleeping { 0.5 } else { 1.0 };

        self.world.player_hunger += sleep_coeff * decay_coeff;
        if self.world.player_hunger > 25.0 {
            if not_boss {
                self.set_inventory_number("food", (self.inventory_f64("food")? - 1.0).max(0.0))?;
            }
            self.world.player_hunger = 0.0;
        }

        self.world.player_thirst += sleep_coeff * decay_coeff;
        if self.world.player_thirst > 20.0 {
            if not_boss {
                self.set_inventory_number("drink", (self.inventory_f64("drink")? - 1.0).max(0.0))?;
            }
            self.world.player_thirst = 0.0;
        }

        self.world.player_fatigue = if self.world.is_sleeping {
            (self.world.player_fatigue - 1.0).min(0.0)
        } else {
            self.world.player_fatigue + decay_coeff
        };
        if self.world.player_fatigue > 30.0 {
            if not_boss {
                self.set_inventory_number(
                    "energy",
                    (self.inventory_f64("energy")? - 1.0).max(0.0),
                )?;
            }
            self.world.player_fatigue = 0.0;
        }
        if self.world.player_fatigue < -10.0 {
            self.set_inventory_number(
                "energy",
                (self.inventory_f64("energy")? + 1.0).min(max_energy),
            )?;
            self.world.player_fatigue = 0.0;
        }

        let has_necessities = self.inventory_f64("food")? > 0.0
            && self.inventory_f64("drink")? > 0.0
            && (self.inventory_f64("energy")? > 0.0 || self.world.is_sleeping);
        self.world.player_recover += if has_necessities {
            if self.world.is_sleeping {
                2.0
            } else {
                1.0
            }
        } else if not_boss {
            if self.world.is_sleeping {
                -0.5
            } else {
                -1.0
            }
        } else {
            0.0
        };
        if self.world.player_recover > 25.0 {
            self.set_inventory_number(
                "health",
                (self.inventory_f64("health")? + 1.0).min(max_health),
            )?;
            self.world.player_recover = 0.0;
        }
        if self.world.player_recover < -15.0 {
            self.set_inventory_number("health", self.inventory_f64("health")? - 1.0)?;
            self.world.player_recover = 0.0;
        }

        let mana_coeff = 1.0 + 0.25 * (self.inventory_i64("intelligence")? - 1) as f64;
        self.world.player_recover_mana = (self.world.player_recover_mana
            + if self.world.is_sleeping { 2.0 } else { 1.0 })
            * mana_coeff;
        if self.world.player_recover_mana > 30.0 {
            self.set_inventory_number("mana", (self.inventory_f64("mana")? + 1.0).min(max_mana))?;
            self.world.player_recover_mana = 0.0;
        }
        Ok(())
    }

    fn newly_unlocked(&self, before: &Value) -> Vec<String> {
        let before_obj = before.as_object();
        let mut names: Vec<String> = self
            .world
            .achievements
            .as_object()
            .into_iter()
            .flat_map(|object| object.iter())
            .filter_map(|(name, count)| {
                let before_count = before_obj
                    .and_then(|object| object.get(name))
                    .and_then(Value::as_i64)
                    .unwrap_or(0);
                if count.as_i64().unwrap_or(0) > 0 && before_count <= 0 {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
    }

    fn unlock(&mut self, achievement: &str) -> Result<()> {
        let object = self
            .world
            .achievements
            .as_object_mut()
            .ok_or_else(|| anyhow!("achievements must be object"))?;
        if object.get(achievement).and_then(Value::as_i64).unwrap_or(0) <= 0 {
            object.insert(achievement.to_string(), json!(1));
            let reward = achievement_reward(achievement);
            self.add_private_f64("reward_last", reward)?;
            self.add_private_f64("total_reward", reward)?;
        }
        Ok(())
    }

    fn reject(&mut self, action: &str, code: &str) -> Result<()> {
        self.add_private_i64("invalid_action_count", 1)?;
        let penalty = self
            .resolved
            .rules
            .get("invalid_action_penalty")
            .and_then(Value::as_f64)
            .unwrap_or(-0.05);
        self.add_private_f64("reward_last", penalty)?;
        self.add_private_f64("total_reward", penalty)?;
        self.events.push(EventRecord {
            step_index: self.world.timestep,
            tick: self.world.timestep,
            episode_id: self.resolved.episode_id.clone(),
            kind: "rule_violation".to_string(),
            action: Some(Value::String(action.to_string())),
            transition: Some("reject".to_string()),
            severity: "warn".to_string(),
            message: format!("RuleViolation({code})"),
            payload: json!({"code": code, "action": action}),
        });
        Ok(())
    }

    /// Report what the recipe cost, what is held, and what is short.
    ///
    /// A bare "missing_resources" leaves the policy to rediscover the recipe
    /// from a system prompt that is thousands of tokens behind it.  Used by
    /// both the crafting and the placement path so they cannot drift.
    fn append_missing_resources(&mut self, action: &str, costs: &[(&str, i64)]) -> Result<()> {
        let mut held = Map::new();
        let mut short = Map::new();
        for (key, amount) in costs {
            let have = self.inventory_i64(key)?;
            held.insert((*key).to_string(), json!(have));
            if have < *amount {
                short.insert((*key).to_string(), json!(*amount - have));
            }
        }
        self.append_noop(
            action,
            "missing_resources",
            json!({
                "costs": costs_json(costs),
                "held": Value::Object(held),
                "short": Value::Object(short),
            }),
        );
        Ok(())
    }

    fn pay(&mut self, costs: &[(&str, i64)]) -> Result<bool> {
        for (key, amount) in costs {
            if self.inventory_i64(key)? < *amount {
                return Ok(false);
            }
        }
        for (key, amount) in costs {
            self.add_inventory_i64(key, -*amount)?;
        }
        Ok(true)
    }

    fn inventory_i64(&self, key: &str) -> Result<i64> {
        let Some(value) = self.world.inventory.get(key) else {
            return Ok(0);
        };
        if let Some(number) = value.as_i64() {
            return Ok(number);
        }
        Ok(value.as_f64().unwrap_or(0.0) as i64)
    }

    fn inventory_f64(&self, key: &str) -> Result<f64> {
        Ok(self
            .world
            .inventory
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or(0.0))
    }

    fn set_inventory_i64(&mut self, key: &str, value: i64) -> Result<()> {
        self.set_inventory_value(key, json!(value))
    }

    fn add_inventory_i64(&mut self, key: &str, delta: i64) -> Result<()> {
        let value = self.inventory_i64(key)? + delta;
        self.set_inventory_i64(key, value)
    }

    fn set_inventory_value(&mut self, key: &str, value: Value) -> Result<()> {
        let object = self
            .world
            .inventory
            .as_object_mut()
            .ok_or_else(|| anyhow!("inventory must be object"))?;
        object.insert(key.to_string(), value);
        Ok(())
    }

    fn set_inventory_number(&mut self, key: &str, value: f64) -> Result<()> {
        let number =
            Number::from_f64(value).ok_or_else(|| anyhow!("inventory.{key} must be finite"))?;
        self.set_inventory_value(key, Value::Number(number))
    }

    fn day_length(&self) -> i64 {
        self.resolved
            .world
            .get("day_length")
            .or_else(|| self.resolved.rules.get("day_length"))
            .and_then(Value::as_i64)
            .unwrap_or(300)
            .max(1)
    }

    fn mob_capacity(&self, class_name: &str) -> i64 {
        let default = match class_name {
            "passive" | "melee" => 3,
            "ranged" => 2,
            _ => 0,
        };
        self.resolved
            .world
            .get(&format!("max_{class_name}_mobs"))
            .and_then(Value::as_i64)
            .unwrap_or(default)
    }

    fn active_mob_count(&self, class_name: &str, level: i64) -> i64 {
        self.world
            .entities
            .iter()
            .filter(|entity| entity.mask && entity.level == level && entity.mob_class == class_name)
            .count() as i64
    }

    fn mob_despawn_distance(&self) -> i64 {
        self.resolved
            .world
            .get("mob_despawn_distance")
            .and_then(Value::as_i64)
            .unwrap_or(14)
    }

    fn private_f64(&self, key: &str) -> Result<f64> {
        self.private
            .get(key)
            .and_then(Value::as_f64)
            .ok_or_else(|| anyhow!("private.{key} must be number"))
    }

    fn set_private_f64(&mut self, key: &str, value: f64) -> Result<()> {
        let number =
            Number::from_f64(value).ok_or_else(|| anyhow!("private.{key} must be finite"))?;
        set_key(&mut self.private, key, Value::Number(number))
    }

    fn add_private_f64(&mut self, key: &str, delta: f64) -> Result<()> {
        let value = self.private_f64(key)? + delta;
        self.set_private_f64(key, value)
    }

    fn set_private_i64(&mut self, key: &str, value: i64) -> Result<()> {
        set_key(&mut self.private, key, json!(value))
    }

    fn add_private_i64(&mut self, key: &str, delta: i64) -> Result<()> {
        let value = self.private.get(key).and_then(Value::as_i64).unwrap_or(0) + delta;
        self.set_private_i64(key, value)
    }

    fn add_private_achievement(&mut self, achievement: &str) -> Result<()> {
        let array = self
            .private
            .get_mut("achievements")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| anyhow!("private.achievements must be array"))?;
        if !array
            .iter()
            .any(|value| value.as_str() == Some(achievement))
        {
            array.push(Value::String(achievement.to_string()));
            array.sort_by(|left, right| left.as_str().cmp(&right.as_str()));
        }
        Ok(())
    }
}

pub fn run_entry(entry: &Value) -> Result<Value> {
    let mut session = CraftaxRustSession::reset_from_entry(entry)?;
    let mut checkpoint_blob: Option<Vec<u8>> = None;
    if let Some(actions) = entry.get("actions").and_then(Value::as_array) {
        let checkpoint_after = entry.get("checkpoint_after").and_then(Value::as_i64);
        for (index, action) in actions.iter().enumerate() {
            if session.is_done() {
                break;
            }
            session.step(action)?;
            if checkpoint_after == Some(index as i64 + 1) {
                checkpoint_blob = Some(session.checkpoint_bytes()?);
            }
        }
    }
    if let Some(blob) = checkpoint_blob {
        session.restore_checkpoint_bytes(&blob)?;
    }
    if let Some(actions) = entry.get("restore_then_actions").and_then(Value::as_array) {
        for action in actions {
            if session.is_done() {
                break;
            }
            session.step(action)?;
        }
    }
    Ok(session.scenario_output())
}

pub fn scenario_to_task(entry: &Value) -> Result<Value> {
    let object = entry
        .as_object()
        .ok_or_else(|| anyhow!("scenario entry must be a JSON object"))?;
    let scenario_id = object
        .get("scenario_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("scenario_id is required"))?;
    Ok(json!({
        "schema": TASK_SCHEMA,
        "task_id": object.get("task_id").and_then(Value::as_str).unwrap_or(scenario_id),
        "scenario_id": scenario_id,
        "seed": object.get("seed").cloned().unwrap_or_else(|| json!(0)),
        // Reference Craftax unless the scenario names another board. The dev
        // board is a fast smoke fixture, not a default anyone should land on
        // by omission.
        "world": object.get("world").cloned().unwrap_or_else(|| json!({"use_default": "craftax_default"})),
        "rules": object.get("rules").cloned().unwrap_or_else(|| json!({"base": "symbolic_no_homeostasis"})),
        "readouts": object.get("readouts").cloned().unwrap_or_else(|| json!({"profile": "symbolic_compact"})),
    }))
}

pub fn resolve_task(task: &Value, seed_override: Option<i64>) -> Result<ResolvedTask> {
    let object = task
        .as_object()
        .ok_or_else(|| anyhow!("task must be a JSON object"))?;
    let schema = object.get("schema").and_then(Value::as_str).unwrap_or("");
    if !schema.is_empty() && schema != TASK_SCHEMA {
        bail!("unsupported Craftax task schema: {schema}");
    }

    let task_id = string_or(object.get("task_id"), None)
        .or_else(|| string_or(object.get("scenario_id"), None))
        .unwrap_or_else(|| "manual".to_string());
    let scenario_id = string_or(object.get("scenario_id"), None).unwrap_or_else(|| task_id.clone());
    let raw_world = object.get("world").cloned().unwrap_or_else(|| json!({}));
    let mut world = merge_world(&raw_world)?;
    let mut rules = merge_rules(&object.get("rules").cloned().unwrap_or_else(|| json!({})))?;
    let readouts = merge_readouts(&object.get("readouts").cloned().unwrap_or_else(|| json!({})))?;

    if let Some(max_steps) = rules.get("max_steps").cloned() {
        set_key(
            &mut rules,
            "max_steps",
            json!(strict_positive_int(&max_steps, "rules.max_steps")?),
        )?;
    }
    let task_max_steps = if let Some(max_steps) = object.get("max_steps") {
        strict_positive_int(max_steps, "max_steps")?
    } else {
        200
    };
    let seed_source = if let Some(seed) = seed_override {
        json!(seed)
    } else if let Some(seed) = world.get("seed") {
        seed.clone()
    } else {
        object.get("seed").cloned().unwrap_or_else(|| json!(0))
    };
    let seed = strict_int(&seed_source, "seed")?;
    set_key(&mut world, "seed", json!(seed))?;

    let width = strict_int(world.get("width").unwrap_or(&json!(48)), "world.width")?;
    let height = strict_int(world.get("height").unwrap_or(&json!(48)), "world.height")?;
    let levels = strict_int(world.get("levels").unwrap_or(&json!(9)), "world.levels")?;
    let max_steps_source = world
        .get("max_steps")
        .cloned()
        .or_else(|| rules.get("max_steps").cloned())
        .unwrap_or_else(|| json!(task_max_steps));
    let max_steps = strict_positive_int(&max_steps_source, "max_steps")?;
    if width < 5 || height < 5 {
        bail!("Craftax world must be at least 5x5");
    }
    if levels <= 0 {
        bail!("Craftax world.levels must be positive");
    }
    set_key(&mut world, "width", json!(width))?;
    set_key(&mut world, "height", json!(height))?;
    set_key(&mut world, "levels", json!(levels))?;
    set_key(&mut world, "max_steps", json!(max_steps))?;

    canonicalize_nonnegative_int(&mut world, "view_radius", "world.view_radius")?;
    canonicalize_nonnegative_int(
        &mut world,
        "checkpoint_every_n_steps",
        "world.checkpoint_every_n_steps",
    )?;
    canonicalize_positive_int(&mut world, "day_length", "world.day_length")?;
    canonicalize_nonnegative_int(
        &mut world,
        "max_player_projectiles",
        "world.max_player_projectiles",
    )?;
    canonicalize_nonnegative_int(
        &mut world,
        "max_mob_projectiles",
        "world.max_mob_projectiles",
    )?;
    canonicalize_nonnegative_int(&mut world, "max_passive_mobs", "world.max_passive_mobs")?;
    canonicalize_nonnegative_int(&mut world, "max_melee_mobs", "world.max_melee_mobs")?;
    canonicalize_nonnegative_int(&mut world, "max_ranged_mobs", "world.max_ranged_mobs")?;
    canonicalize_nonnegative_int(
        &mut world,
        "mob_despawn_distance",
        "world.mob_despawn_distance",
    )?;
    canonicalize_positive_int(&mut rules, "day_length", "rules.day_length")?;
    canonicalize_finite_number(&mut rules, "achievement_reward", "rules.achievement_reward")?;
    canonicalize_finite_number(&mut rules, "step_reward", "rules.step_reward")?;
    canonicalize_finite_number(
        &mut rules,
        "invalid_action_penalty",
        "rules.invalid_action_penalty",
    )?;
    canonicalize_finite_number(&mut rules, "death_penalty", "rules.death_penalty")?;
    canonicalize_bool(&mut rules, "homeostasis", "rules.homeostasis")?;
    canonicalize_bool(&mut rules, "god_mode", "rules.god_mode")?;

    let expanded = json!({
        "task_id": task_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "width": width,
        "height": height,
        "max_steps": max_steps,
        "world": world,
        "rules": rules,
        "readouts": readouts,
    });
    let config_hash = stable_hash(&expanded, 16)?;
    let episode_id = stable_hash(
        &Value::String(format!(
            "gamebench.craftax-singleplayer.episode:{task_id}:{seed}:{config_hash}"
        )),
        32,
    )?;
    Ok(ResolvedTask {
        task_id,
        scenario_id,
        seed,
        width,
        height,
        max_steps,
        world: expanded["world"].clone(),
        rules: expanded["rules"].clone(),
        readouts: expanded["readouts"].clone(),
        config_hash,
        episode_id,
    })
}

pub fn stable_hash(value: &Value, length: usize) -> Result<String> {
    let encoded = serde_json::to_vec(value).context("encode canonical JSON for stable hash")?;
    let digest = Sha256::digest(encoded);
    Ok(format!("{digest:x}").chars().take(length).collect())
}

fn merge_world(raw: &Value) -> Result<Value> {
    let raw_object = as_object(raw, "world")?;
    let mut base = if let Some(default_name) = raw_object.get("use_default").and_then(Value::as_str)
    {
        let defaults = load_json(&task_dir().join("defaults/worlds.json"))?;
        defaults
            .get(default_name)
            .cloned()
            .ok_or_else(|| anyhow!("unknown Craftax world default: {default_name}"))?
    } else {
        json!({})
    };
    let mut patch = Map::new();
    for (key, value) in raw_object {
        if key != "use_default" {
            patch.insert(key.clone(), value.clone());
        }
    }
    base = deep_merge(&base, &Value::Object(patch))?;
    set_default(&mut base, "width", json!(48))?;
    set_default(&mut base, "height", json!(48))?;
    set_default(&mut base, "view_radius", json!(4))?;
    set_default(&mut base, "levels", json!(9))?;
    set_default(&mut base, "densities", json!({}))?;
    Ok(base)
}

fn merge_rules(raw: &Value) -> Result<Value> {
    let raw_object = as_object(raw, "rules")?;
    let mut base = if let Some(base_name) = raw_object.get("base").and_then(Value::as_str) {
        let path = task_dir().join(format!("defaults/rules/{base_name}.json"));
        if path.exists() {
            load_json(&path)?
        } else {
            json!({"base": base_name})
        }
    } else {
        load_json(&task_dir().join("defaults/rules/symbolic_survival.json"))?
    };
    let overrides = raw_object
        .get("overrides")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut patch = Map::new();
    for (key, value) in raw_object {
        if key != "base" && key != "overrides" {
            patch.insert(key.clone(), value.clone());
        }
    }
    base = deep_merge(&base, &Value::Object(patch))?;
    base = deep_merge(&base, &overrides)?;
    set_default(&mut base, "achievement_reward", json!(1.0))?;
    set_default(&mut base, "step_reward", json!(0.0))?;
    set_default(&mut base, "invalid_action_penalty", json!(-0.05))?;
    set_default(&mut base, "death_penalty", json!(-1.0))?;
    set_default(&mut base, "homeostasis", json!(true))?;
    Ok(base)
}

fn merge_readouts(raw: &Value) -> Result<Value> {
    let raw_object = as_object(raw, "readouts")?;
    let mut base = json!({
        "symbolic": true,
        "local_map": true,
        "full_world_state": false,
        "observation_text": true,
    });
    if let Some(profile) = raw_object.get("profile").and_then(Value::as_str) {
        let defaults = load_json(&task_dir().join("defaults/readouts.json"))?;
        if let Some(profile_defaults) = defaults.get(profile) {
            base = deep_merge(&base, profile_defaults)?;
        }
    }
    let mut patch = Map::new();
    for (key, value) in raw_object {
        if key != "profile" {
            patch.insert(key.clone(), value.clone());
        }
    }
    deep_merge(&base, &Value::Object(patch))
}

fn deep_merge(base: &Value, patch: &Value) -> Result<Value> {
    match (base, patch) {
        (Value::Object(base_map), Value::Object(patch_map)) => {
            let mut merged = base_map.clone();
            for (key, value) in patch_map {
                let next = if let Some(existing) = merged.get(key) {
                    deep_merge(existing, value)?
                } else {
                    value.clone()
                };
                merged.insert(key.clone(), next);
            }
            Ok(Value::Object(merged))
        }
        (_, _) => Ok(patch.clone()),
    }
}

fn load_json(path: &Path) -> Result<Value> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

fn task_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("gold_rust has task directory parent")
        .to_path_buf()
}

fn default_inventory() -> Value {
    json!({
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
    })
}

fn as_object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| anyhow!("Craftax {field} must be object: {value:?}"))
}

fn string_or(value: Option<&Value>, default: Option<&str>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| default.map(ToString::to_string))
}

fn strict_int(value: &Value, field: &str) -> Result<i64> {
    value
        .as_i64()
        .filter(|_| !value.is_boolean())
        .ok_or_else(|| anyhow!("Craftax {field} must be integer: {value:?}"))
}

fn strict_positive_int(value: &Value, field: &str) -> Result<i64> {
    let number = strict_int(value, field)?;
    if number <= 0 {
        bail!("Craftax {field} must be positive: {value:?}");
    }
    Ok(number)
}

fn strict_nonnegative_int(value: &Value, field: &str) -> Result<i64> {
    let number = strict_int(value, field)?;
    if number < 0 {
        bail!("Craftax {field} must be nonnegative: {value:?}");
    }
    Ok(number)
}

fn strict_finite_number(value: &Value, field: &str) -> Result<f64> {
    let number = value
        .as_f64()
        .filter(|number| number.is_finite())
        .ok_or_else(|| anyhow!("Craftax {field} must be finite number: {value:?}"))?;
    Ok(number)
}

fn strict_bool(value: &Value, field: &str) -> Result<bool> {
    value
        .as_bool()
        .ok_or_else(|| anyhow!("Craftax {field} must be boolean: {value:?}"))
}

fn set_default(mapping: &mut Value, key: &str, value: Value) -> Result<()> {
    let object = mapping
        .as_object_mut()
        .ok_or_else(|| anyhow!("expected object while setting default {key}"))?;
    object.entry(key.to_string()).or_insert(value);
    Ok(())
}

fn set_key(mapping: &mut Value, key: &str, value: Value) -> Result<()> {
    let object = mapping
        .as_object_mut()
        .ok_or_else(|| anyhow!("expected object while setting {key}"))?;
    object.insert(key.to_string(), value);
    Ok(())
}

fn canonicalize_positive_int(mapping: &mut Value, key: &str, field: &str) -> Result<()> {
    if let Some(value) = mapping.get(key).cloned() {
        set_key(mapping, key, json!(strict_positive_int(&value, field)?))?;
    }
    Ok(())
}

fn canonicalize_nonnegative_int(mapping: &mut Value, key: &str, field: &str) -> Result<()> {
    if let Some(value) = mapping.get(key).cloned() {
        set_key(mapping, key, json!(strict_nonnegative_int(&value, field)?))?;
    }
    Ok(())
}

fn canonicalize_finite_number(mapping: &mut Value, key: &str, field: &str) -> Result<()> {
    if let Some(value) = mapping.get(key).cloned() {
        let number = Number::from_f64(strict_finite_number(&value, field)?)
            .ok_or_else(|| anyhow!("Craftax {field} must be finite number: {value:?}"))?;
        set_key(mapping, key, Value::Number(number))?;
    }
    Ok(())
}

fn canonicalize_bool(mapping: &mut Value, key: &str, field: &str) -> Result<()> {
    if let Some(value) = mapping.get(key).cloned() {
        set_key(mapping, key, json!(strict_bool(&value, field)?))?;
    }
    Ok(())
}

fn make_world(resolved: &ResolvedTask, rng: &mut PythonRandom) -> Result<CraftaxWorld> {
    let width = resolved.width;
    let height = resolved.height;
    let levels = resolved
        .world
        .get("levels")
        .map(|value| strict_int(value, "world.levels"))
        .transpose()?
        .unwrap_or(9);
    let layout = generate_world_layout(width, height, levels, rng, resolved.world.get("densities"));
    let mut monsters_killed = vec![0; usize::try_from(levels).context("levels must fit usize")?];
    if !monsters_killed.is_empty() {
        monsters_killed[0] = 10;
    }
    let mut potion_mapping = vec![0, 1, 2, 3, 4, 5];
    rng.shuffle(&mut potion_mapping);
    let mut world = CraftaxWorld {
        width,
        height,
        levels,
        max_steps: resolved.max_steps,
        seed: resolved.seed,
        maps: layout.maps,
        item_maps: layout.item_maps,
        down_ladders: layout.down_ladders,
        up_ladders: layout.up_ladders,
        chests_opened: vec![false; usize::try_from(levels).context("levels must fit usize")?],
        monsters_killed,
        potion_mapping,
        player_pos: layout.player_pos,
        player_direction: (0, 1),
        player_level: 0,
        timestep: 0,
        light_level: calculate_light_level(0, day_length_from_resolved(resolved)),
        is_sleeping: false,
        is_resting: false,
        player_recover: 0.0,
        player_hunger: 0.0,
        player_thirst: 0.0,
        player_fatigue: 0.0,
        player_recover_mana: 0.0,
        inventory: default_inventory(),
        achievements: json!({}),
        entities: Vec::new(),
        player_projectiles: Vec::new(),
        mob_projectiles: Vec::new(),
    };
    world.entities = spawn_entities(&world, rng)?;
    apply_initial_state(&mut world, resolved.world.get("initial_state"))?;
    Ok(world)
}

struct WorldLayout {
    maps: Vec<Vec<Vec<String>>>,
    item_maps: Vec<Vec<Vec<String>>>,
    down_ladders: Vec<(i64, i64)>,
    up_ladders: Vec<(i64, i64)>,
    player_pos: (i64, i64),
}

fn generate_world_layout(
    width: i64,
    height: i64,
    levels: i64,
    rng: &mut PythonRandom,
    densities: Option<&Value>,
) -> WorldLayout {
    let player_pos = (width / 2, height / 2);
    let mut maps = Vec::new();
    let mut item_maps = Vec::new();
    let mut down_ladders = Vec::new();
    let mut up_ladders = Vec::new();
    for level in 0..levels {
        let (grid, items, down_ladder, up_ladder) = if level == 1 || level == 3 || level == 4 {
            generate_small_dungeon(width, height, rng, level)
        } else {
            generate_smooth_level(width, height, rng, player_pos, level, densities)
        };
        maps.push(grid);
        item_maps.push(items);
        down_ladders.push(down_ladder);
        up_ladders.push(up_ladder);
    }
    WorldLayout {
        maps,
        item_maps,
        down_ladders,
        up_ladders,
        player_pos,
    }
}

fn generate_smooth_level(
    width: i64,
    height: i64,
    rng: &mut PythonRandom,
    player_pos: (i64, i64),
    level: i64,
    densities: Option<&Value>,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, (i64, i64), (i64, i64)) {
    let config = smooth_config(level);
    let larger_res = (std::cmp::max(1, width / 4), std::cmp::max(1, height / 4));
    let small_res = (std::cmp::max(1, width / 16), std::cmp::max(1, height / 16));
    let x_res = (std::cmp::max(1, width / 8), std::cmp::max(1, height / 2));
    let water_noise = fractal_noise_2d(rng, width, height, small_res);
    let mountain_noise = fractal_noise_2d(rng, width, height, small_res);
    let path_x = fractal_noise_2d(rng, width, height, x_res);
    let tree_noise = fractal_noise_2d(rng, width, height, larger_res);
    let mut grid = vec![
        vec![config.default_block.to_string(); usize::try_from(width).unwrap()];
        usize::try_from(height).unwrap()
    ];
    let water_density = density(densities, "water", 1.0);
    let tree_density = density(densities, "tree", 1.0);
    let mountain_threshold = 0.7;
    for y in 0..height {
        for x in 0..width {
            let distance = (((x - player_pos.0).pow(2) + (y - player_pos.1).pow(2)) as f64).sqrt();
            let water_proximity = config
                .water_max
                .min(distance / std::cmp::max(1, config.water_strength) as f64);
            let mut water = water_noise[y as usize][x as usize] + water_proximity - 1.0;
            if water_density <= 0.0 {
                water = -1.0;
            }
            let (water_cut, sand_cut) =
                water_cuts(config.water_threshold, config.sand_threshold, water_density);
            let mut block = if water > water_cut {
                config.sea_block
            } else {
                config.default_block
            };
            if water > sand_cut && block != config.sea_block {
                block = config.coast_block;
            }

            let mountain_proximity = config
                .mountain_max
                .min(distance / std::cmp::max(1, config.mountain_strength) as f64);
            let mountain = mountain_noise[y as usize][x as usize] + 0.05 + mountain_proximity - 1.0;
            if mountain > mountain_threshold {
                block = config.mountain_block;
            }
            let path_y = if x < height && y < width {
                path_x[x as usize][y as usize]
            } else {
                path_x[y as usize][x as usize]
            };
            if mountain > mountain_threshold
                && (path_x[y as usize][x as usize] > 0.8 || path_y > 0.8)
            {
                block = config.path_block;
            }
            if mountain > 0.85 && water > 0.4 {
                block = config.inner_mountain_block;
            }
            if tree_density > 0.0
                && block == config.tree_requirement_block
                && tree_noise[y as usize][x as usize] > config.tree_threshold_perlin
                && rng.random()
                    > tree_uniform_threshold(config.tree_threshold_uniform, tree_density)
            {
                block = config.tree;
            }
            grid[y as usize][x as usize] = block.to_string();
        }
    }
    for y in 0..height {
        for x in 0..width {
            if mountain_noise[y as usize][x as usize] + 0.05 > 0.85
                && tree_noise[y as usize][x as usize] > 0.7
            {
                let current = grid[y as usize][x as usize].as_str();
                if current == config.mountain_block
                    || current == config.inner_mountain_block
                    || current == config.tree
                {
                    grid[y as usize][x as usize] = config.lava.to_string();
                }
            }
        }
    }
    for &(req, ore, chance) in config.ores {
        if chance <= 0.0 || ore == "out_of_bounds" {
            continue;
        }
        for y in 0..height {
            for x in 0..width {
                if grid[y as usize][x as usize] == req && rng.random() < chance {
                    grid[y as usize][x as usize] = ore.to_string();
                }
            }
        }
    }
    if player_pos.0 >= 0 && player_pos.0 < width && player_pos.1 >= 0 && player_pos.1 < height {
        grid[player_pos.1 as usize][player_pos.0 as usize] = config.player_spawn.to_string();
    }
    let mut item_map = empty_item_map(width, height);
    let down_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, player_pos);
    let up_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, down_ladder);
    if config.ladder_down {
        item_map[down_ladder.1 as usize][down_ladder.0 as usize] = "ladder_down".to_string();
    }
    if config.ladder_up {
        item_map[up_ladder.1 as usize][up_ladder.0 as usize] = "ladder_up".to_string();
    }
    (grid, item_map, down_ladder, up_ladder)
}

fn generate_small_dungeon(
    width: i64,
    height: i64,
    rng: &mut PythonRandom,
    level: i64,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, (i64, i64), (i64, i64)) {
    if width >= 10 && height >= 10 {
        return generate_full_dungeon(width, height, rng, level);
    }
    let config = dungeon_config(level);
    let mut grid = vec![vec!["wall".to_string(); width as usize]; height as usize];
    let mut item_map = empty_item_map(width, height);
    for y in 1..std::cmp::max(1, height - 1) {
        for x in 1..std::cmp::max(1, width - 1) {
            grid[y as usize][x as usize] = "path".to_string();
        }
    }
    if width > 4 && height > 4 {
        grid[2][2] = config.special_block.to_string();
    }
    let down_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, (-1, -1));
    item_map[down_ladder.1 as usize][down_ladder.0 as usize] = "ladder_down".to_string();
    let up_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, down_ladder);
    item_map[up_ladder.1 as usize][up_ladder.0 as usize] = "ladder_up".to_string();
    (grid, item_map, down_ladder, up_ladder)
}

fn generate_full_dungeon(
    width: i64,
    height: i64,
    rng: &mut PythonRandom,
    level: i64,
) -> (Vec<Vec<String>>, Vec<Vec<String>>, (i64, i64), (i64, i64)) {
    let config = dungeon_config(level);
    let mut grid = vec![vec!["wall".to_string(); width as usize]; height as usize];
    let mut item_map = empty_item_map(width, height);
    let chunks_x = std::cmp::max(1, width / 16);
    let chunks_y = std::cmp::max(1, height / 16);
    let mut chunks = Vec::new();
    for cy in 0..chunks_y {
        for cx in 0..chunks_x {
            chunks.push((cx, cy));
        }
    }
    rng.shuffle(&mut chunks);

    let mut rooms: Vec<(i64, i64, i64, i64)> = Vec::new();
    for room_index in 0..std::cmp::min(8, chunks.len()) {
        let (cx, cy) = chunks[room_index];
        let mut room_w = rng.randrange(5, 10);
        let mut room_h = rng.randrange(5, 10);
        let origin_x = cx * 16 + rng.randrange(0, std::cmp::max(1, 16 - 5));
        let origin_y = cy * 16 + rng.randrange(0, std::cmp::max(1, 16 - 5));
        let x0 = std::cmp::min(
            std::cmp::max(1, origin_x),
            std::cmp::max(1, width - room_w - 1),
        );
        let y0 = std::cmp::min(
            std::cmp::max(1, origin_y),
            std::cmp::max(1, height - room_h - 1),
        );
        room_w = std::cmp::min(room_w, width - x0 - 1);
        room_h = std::cmp::min(room_h, height - y0 - 1);
        rooms.push((x0, y0, room_w, room_h));
        for y in y0..(y0 + room_h) {
            for x in x0..(x0 + room_w) {
                grid[y as usize][x as usize] = "path".to_string();
            }
        }
        for (tx, ty) in [
            (x0, y0),
            (x0 + room_w - 1, y0),
            (x0, y0 + room_h - 1),
            (x0 + room_w - 1, y0 + room_h - 1),
        ] {
            item_map[ty as usize][tx as usize] = "torch".to_string();
        }
        if room_w > 2 && room_h > 2 {
            let chest = (
                rng.randrange(x0 + 1, x0 + room_w - 1),
                rng.randrange(y0 + 1, y0 + room_h - 1),
            );
            grid[chest.1 as usize][chest.0 as usize] = "chest".to_string();
            if rng.random() > 0.5 {
                let fountain = (
                    rng.randrange(x0 + 1, x0 + room_w - 1),
                    rng.randrange(y0 + 1, y0 + room_h - 1),
                );
                grid[fountain.1 as usize][fountain.0 as usize] = config.fountain_block.to_string();
            }
        }
    }

    if rooms.is_empty() {
        return generate_small_dungeon(width, height, rng, level);
    }
    let mut included = vec![*rooms.last().expect("rooms not empty")];
    for room in &rooms {
        let sink = included[rng.randrange(0, included.len() as i64) as usize];
        carve_corridor(&mut grid, room_anchor(*room), room_anchor(sink));
        if !included.contains(room) {
            included.push(*room);
        }
    }
    let special_room = rooms[0];
    let sx = std::cmp::min(width - 2, special_room.0 + 2);
    let sy = std::cmp::min(height - 2, special_room.1 + 2);
    grid[sy as usize][sx as usize] = config.special_block.to_string();

    grid = apply_dungeon_visuals(&grid, &item_map, rng, config);
    let down_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, (-1, -1));
    item_map[down_ladder.1 as usize][down_ladder.0 as usize] = "ladder_down".to_string();
    let up_ladder = choose_ladder(&grid, &item_map, config.valid_ladder, rng, down_ladder);
    item_map[up_ladder.1 as usize][up_ladder.0 as usize] = "ladder_up".to_string();
    (grid, item_map, down_ladder, up_ladder)
}

fn apply_dungeon_visuals(
    grid: &[Vec<String>],
    item_map: &[Vec<String>],
    rng: &mut PythonRandom,
    config: DungeonConfigRust,
) -> Vec<Vec<String>> {
    let height = grid.len();
    let width = grid.first().map(Vec::len).unwrap_or(0);
    let mut out = grid.to_vec();
    for y in 0..height {
        for x in 0..width {
            let adjacent_path = has_cross_neighbor(grid, x as i64, y as i64);
            let rare = rng.random() < 0.1;
            if grid[y][x] == "wall" {
                out[y][x] = if adjacent_path && rare {
                    "wall_moss".to_string()
                } else if adjacent_path {
                    "wall".to_string()
                } else {
                    "darkness".to_string()
                };
            } else if rare && grid[y][x] == "path" && item_map[y][x] == "none" {
                out[y][x] = config.rare_path_replacement_block.to_string();
            }
        }
    }
    out
}

fn has_cross_neighbor(grid: &[Vec<String>], x: i64, y: i64) -> bool {
    let height = grid.len() as i64;
    let width = grid.first().map(Vec::len).unwrap_or(0) as i64;
    for (nx, ny) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
        if nx >= 0
            && nx < width
            && ny >= 0
            && ny < height
            && grid[ny as usize][nx as usize] != "wall"
        {
            return true;
        }
    }
    false
}

fn carve_corridor(grid: &mut [Vec<String>], source: (i64, i64), sink: (i64, i64)) {
    let (sx, sy) = source;
    let (tx, ty) = sink;
    let step_x = if tx >= sx { 1 } else { -1 };
    let mut x = sx;
    loop {
        if grid[sy as usize][x as usize] == "wall" {
            grid[sy as usize][x as usize] = "path".to_string();
        }
        if x == tx {
            break;
        }
        x += step_x;
    }
    let step_y = if ty >= sy { 1 } else { -1 };
    let mut y = sy;
    loop {
        if grid[y as usize][tx as usize] == "wall" {
            grid[y as usize][tx as usize] = "path".to_string();
        }
        if y == ty {
            break;
        }
        y += step_y;
    }
}

fn room_anchor(room: (i64, i64, i64, i64)) -> (i64, i64) {
    (room.0, room.1)
}

fn spawn_entities(world: &CraftaxWorld, rng: &mut PythonRandom) -> Result<Vec<Entity>> {
    let mut entities = Vec::new();
    let mut occupied: HashSet<(i64, i64, i64)> = HashSet::new();
    let per_level = std::cmp::max(2, world.width / 8);
    let mut idx = 0;
    for level in 0..world.levels {
        if level == 8 {
            continue;
        }
        let choices = floor_mobs(level);
        for local_idx in 0..per_level {
            let kind = choices[usize::try_from(local_idx).unwrap() % choices.len()];
            let mob_class = mob_class(kind);
            for _ in 0..80 {
                let pos = (
                    rng.randrange(1, world.width - 1),
                    rng.randrange(1, world.height - 1),
                );
                let block = &world.maps[level as usize][pos.1 as usize][pos.0 as usize];
                if mob_can_occupy_block(kind, mob_class, block)
                    && !occupied.contains(&(level, pos.0, pos.1))
                    && (level != world.player_level || pos != world.player_pos)
                {
                    entities.push(Entity {
                        id: format!("{kind}_{idx}"),
                        kind: kind.to_string(),
                        pos,
                        health: mob_health(kind) as f64,
                        level,
                        mob_class: mob_class.to_string(),
                        attack_cooldown: 0,
                        mask: true,
                    });
                    occupied.insert((level, pos.0, pos.1));
                    idx += 1;
                    break;
                }
            }
        }
    }
    Ok(entities)
}

fn apply_initial_state(world: &mut CraftaxWorld, initial: Option<&Value>) -> Result<()> {
    let Some(initial) = initial else {
        return Ok(());
    };
    let object = as_object(initial, "world.initial_state")?;
    if let Some(map) = object.get("map") {
        let level = object
            .get("map_level")
            .map(|value| strict_int(value, "initial_state.map_level"))
            .transpose()?
            .unwrap_or(world.player_level);
        apply_initial_ascii_map(world, level, map)?;
    }
    if let Some(player) = object.get("player") {
        let player = as_object(player, "initial_state.player")?;
        if let Some(pos) = player.get("pos") {
            world.player_pos = pos2(pos, "player.pos")?;
        }
        if let Some(direction) = player.get("direction").or_else(|| player.get("facing")) {
            world.player_direction = pos2(direction, "player.direction")?;
        }
        if let Some(level) = player.get("level") {
            world.player_level = strict_int(level, "player.level")?;
        }
    }
    if let Some(tiles) = object.get("tiles").and_then(Value::as_array) {
        for tile in tiles {
            let patch = as_object(tile, "initial_state.tiles[]")?;
            let level = patch
                .get("level")
                .map(|value| strict_int(value, "tile.level"))
                .transpose()?
                .unwrap_or(world.player_level);
            let (x, y) = pos2(
                patch
                    .get("pos")
                    .ok_or_else(|| anyhow!("tile.pos is required"))?,
                "tile.pos",
            )?;
            let kind = patch
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("tile.kind is required"))?;
            let normalized = normalize_tile(kind);
            if item_overlay(normalized) || normalized == "none" {
                set_item_cell(world, level, (x, y), normalized);
            } else {
                world.maps[level as usize][y as usize][x as usize] = normalized.to_string();
                if static_solid(normalized) {
                    set_item_cell(world, level, (x, y), "none");
                }
            }
        }
    }
    if let Some(items) = object.get("items").and_then(Value::as_array) {
        for item in items {
            let patch = as_object(item, "initial_state.items[]")?;
            let level = patch
                .get("level")
                .map(|value| strict_int(value, "item.level"))
                .transpose()?
                .unwrap_or(world.player_level);
            let (x, y) = pos2(
                patch
                    .get("pos")
                    .ok_or_else(|| anyhow!("item.pos is required"))?,
                "item.pos",
            )?;
            let kind = patch
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("item.kind is required"))?;
            set_item_cell(world, level, (x, y), kind);
        }
    }
    if let Some(inventory) = object.get("inventory") {
        world.inventory = deep_merge(&default_inventory(), inventory)?;
    }
    if let Some(monsters_killed) = object.get("monsters_killed").and_then(Value::as_array) {
        world.monsters_killed = monsters_killed
            .iter()
            .map(|value| strict_int(value, "monsters_killed[]"))
            .collect::<Result<Vec<_>>>()?;
        world.monsters_killed.resize(world.levels as usize, 0);
    }
    if let Some(chests_opened) = object.get("chests_opened").and_then(Value::as_array) {
        world.chests_opened = chests_opened
            .iter()
            .map(|value| {
                value
                    .as_bool()
                    .ok_or_else(|| anyhow!("chests_opened[] must be bool"))
            })
            .collect::<Result<Vec<_>>>()?;
        world.chests_opened.resize(world.levels as usize, false);
    }
    if let Some(potion_mapping) = object.get("potion_mapping").and_then(Value::as_array) {
        world.potion_mapping = potion_mapping
            .iter()
            .map(|value| strict_int(value, "potion_mapping[]"))
            .collect::<Result<Vec<_>>>()?;
        world.potion_mapping.resize(6, 0);
    }
    if let Some(entities) = object.get("entities").and_then(Value::as_array) {
        world.entities.clear();
        for (idx, item) in entities.iter().enumerate() {
            let data = as_object(item, "initial_state.entities[]")?;
            let kind = normalize_mob(
                data.get("kind")
                    .and_then(Value::as_str)
                    .ok_or_else(|| anyhow!("entity.kind is required"))?,
            );
            let level = data
                .get("level")
                .map(|value| strict_int(value, "entity.level"))
                .transpose()?
                .unwrap_or(world.player_level);
            world.entities.push(Entity {
                id: data
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| format!("{kind}_{idx}")),
                kind: kind.to_string(),
                pos: pos2(
                    data.get("pos")
                        .ok_or_else(|| anyhow!("entity.pos is required"))?,
                    "entity.pos",
                )?,
                health: data
                    .get("health")
                    .and_then(Value::as_f64)
                    .unwrap_or_else(|| mob_health(kind) as f64),
                level,
                mob_class: data
                    .get("class")
                    .or_else(|| data.get("mob_class"))
                    .and_then(Value::as_str)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| mob_class(kind).to_string()),
                attack_cooldown: data
                    .get("attack_cooldown")
                    .map(|value| strict_int(value, "entity.attack_cooldown"))
                    .transpose()?
                    .unwrap_or(0),
                mask: data
                    .get("mask")
                    .map(|value| strict_bool(value, "entity.mask"))
                    .transpose()?
                    .unwrap_or(true),
            });
        }
    }
    if let Some(projectiles) = object.get("player_projectiles").and_then(Value::as_array) {
        world.player_projectiles = parse_projectiles(projectiles, "player", world.player_level)?;
    }
    if let Some(projectiles) = object.get("mob_projectiles").and_then(Value::as_array) {
        world.mob_projectiles = parse_projectiles(projectiles, "mob", world.player_level)?;
    }
    Ok(())
}

fn apply_initial_ascii_map(world: &mut CraftaxWorld, level: i64, map: &Value) -> Result<()> {
    let rows = map
        .as_array()
        .ok_or_else(|| anyhow!("initial_state.map must be an array of strings"))?;
    let level_index = usize::try_from(level).context("initial_state.map_level must fit usize")?;
    if level_index >= world.maps.len() {
        bail!("initial_state.map_level out of bounds: {level}");
    }
    if rows.len() > usize::try_from(world.height).context("world.height must fit usize")? {
        bail!("initial_state.map has more rows than world.height");
    }
    for (y, row_value) in rows.iter().enumerate() {
        let row = row_value
            .as_str()
            .ok_or_else(|| anyhow!("initial_state.map rows must be strings"))?;
        if row.chars().count()
            > usize::try_from(world.width).context("world.width must fit usize")?
        {
            bail!("initial_state.map row {y} is wider than world.width");
        }
        for (x, glyph) in row.chars().enumerate() {
            let pos = (x as i64, y as i64);
            world.item_maps[level_index][y][x] = "none".to_string();
            match glyph {
                '@' => {
                    world.maps[level_index][y][x] = "path".to_string();
                    world.player_pos = pos;
                    world.player_level = level;
                }
                '.' => world.maps[level_index][y][x] = "path".to_string(),
                'g' => world.maps[level_index][y][x] = "grass".to_string(),
                ',' => world.maps[level_index][y][x] = "sand".to_string(),
                '#' => world.maps[level_index][y][x] = "wall".to_string(),
                '~' => world.maps[level_index][y][x] = "water".to_string(),
                'T' => world.maps[level_index][y][x] = "tree".to_string(),
                'S' => world.maps[level_index][y][x] = "stone".to_string(),
                'C' => world.maps[level_index][y][x] = "coal".to_string(),
                'I' => world.maps[level_index][y][x] = "iron".to_string(),
                'D' => world.maps[level_index][y][x] = "diamond".to_string(),
                'A' => world.maps[level_index][y][x] = "crafting_table".to_string(),
                'F' => world.maps[level_index][y][x] = "furnace".to_string(),
                'L' => world.maps[level_index][y][x] = "lava".to_string(),
                'p' => world.maps[level_index][y][x] = "plant".to_string(),
                'R' => world.maps[level_index][y][x] = "ripe_plant".to_string(),
                '>' => {
                    world.maps[level_index][y][x] = "path".to_string();
                    set_item_cell(world, level, pos, "ladder_down");
                }
                '<' => {
                    world.maps[level_index][y][x] = "path".to_string();
                    set_item_cell(world, level, pos, "ladder_up");
                }
                '!' => {
                    world.maps[level_index][y][x] = "path".to_string();
                    set_item_cell(world, level, pos, "torch");
                }
                ' ' => world.maps[level_index][y][x] = "darkness".to_string(),
                other => bail!("unsupported initial_state.map glyph: {other:?}"),
            }
        }
    }
    Ok(())
}

fn parse_projectiles(
    projectiles: &[Value],
    owner: &str,
    default_level: i64,
) -> Result<Vec<Projectile>> {
    let mut parsed = Vec::new();
    for item in projectiles {
        let data = as_object(item, "initial_state.projectiles[]")?;
        parsed.push(Projectile {
            id: data
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("projectile.id is required"))?
                .to_string(),
            kind: data
                .get("kind")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("projectile.kind is required"))?
                .to_string(),
            pos: pos2(
                data.get("pos")
                    .ok_or_else(|| anyhow!("projectile.pos is required"))?,
                "projectile.pos",
            )?,
            direction: pos2(
                data.get("direction")
                    .ok_or_else(|| anyhow!("projectile.direction is required"))?,
                "projectile.direction",
            )?,
            level: data
                .get("level")
                .map(|value| strict_int(value, "projectile.level"))
                .transpose()?
                .unwrap_or(default_level),
            owner: data
                .get("owner")
                .and_then(Value::as_str)
                .unwrap_or(owner)
                .to_string(),
            mask: data
                .get("mask")
                .map(|value| strict_bool(value, "projectile.mask"))
                .transpose()?
                .unwrap_or(true),
        });
    }
    Ok(parsed)
}

#[derive(Clone, Copy)]
struct SmoothConfig {
    default_block: &'static str,
    sea_block: &'static str,
    coast_block: &'static str,
    mountain_block: &'static str,
    path_block: &'static str,
    inner_mountain_block: &'static str,
    ores: &'static [(&'static str, &'static str, f64)],
    tree_requirement_block: &'static str,
    tree: &'static str,
    lava: &'static str,
    player_spawn: &'static str,
    valid_ladder: &'static str,
    ladder_up: bool,
    ladder_down: bool,
    water_strength: i64,
    water_max: f64,
    mountain_strength: i64,
    mountain_max: f64,
    water_threshold: f64,
    sand_threshold: f64,
    tree_threshold_uniform: f64,
    tree_threshold_perlin: f64,
}

#[derive(Clone, Copy)]
struct DungeonConfigRust {
    special_block: &'static str,
    fountain_block: &'static str,
    rare_path_replacement_block: &'static str,
    valid_ladder: &'static str,
}

const OVERWORLD_ORES: &[(&str, &str, f64)] = &[
    ("stone", "coal", 0.03),
    ("stone", "iron", 0.02),
    ("stone", "diamond", 0.001),
    ("stone", "out_of_bounds", 0.0),
    ("stone", "out_of_bounds", 0.0),
];
const MINES_ORES: &[(&str, &str, f64)] = &[
    ("stone", "coal", 0.04),
    ("stone", "iron", 0.02),
    ("stone", "diamond", 0.005),
    ("stone", "sapphire", 0.0025),
    ("stone", "ruby", 0.0025),
];
const TROLL_ORES: &[(&str, &str, f64)] = &[
    ("stone", "coal", 0.04),
    ("stone", "iron", 0.03),
    ("stone", "diamond", 0.01),
    ("stone", "sapphire", 0.01),
    ("stone", "ruby", 0.01),
];
const FIRE_ORES: &[(&str, &str, f64)] = &[
    ("stone", "coal", 0.05),
    ("stone", "iron", 0.0),
    ("stone", "diamond", 0.0),
    ("stone", "sapphire", 0.0),
    ("stone", "ruby", 0.025),
];
const ICE_ORES: &[(&str, &str, f64)] = &[
    ("stone", "coal", 0.0),
    ("stone", "iron", 0.0),
    ("stone", "diamond", 0.005),
    ("stone", "sapphire", 0.02),
    ("stone", "ruby", 0.0),
];
const BOSS_ORES: &[(&str, &str, f64)] = &[
    ("wall", "wall_moss", 0.1),
    ("grave", "grave2", 0.333),
    ("grave", "grave3", 0.5),
    ("wall", "sapphire", 0.0),
    ("wall", "ruby", 0.0),
];

fn smooth_config(level: i64) -> SmoothConfig {
    match level {
        2 => SmoothConfig {
            default_block: "path",
            sea_block: "water",
            coast_block: "path",
            mountain_block: "stone",
            path_block: "stone",
            inner_mountain_block: "stone",
            ores: MINES_ORES,
            tree_requirement_block: "path",
            tree: "stalagmite",
            lava: "lava",
            player_spawn: "path",
            valid_ladder: "path",
            ladder_up: true,
            ladder_down: true,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 17,
            mountain_max: 1.5,
            water_threshold: 0.7,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.8,
            tree_threshold_perlin: 0.5,
        },
        5 => SmoothConfig {
            default_block: "path",
            sea_block: "water",
            coast_block: "path",
            mountain_block: "stone",
            path_block: "stone",
            inner_mountain_block: "stone",
            ores: TROLL_ORES,
            tree_requirement_block: "path",
            tree: "stalagmite",
            lava: "lava",
            player_spawn: "path",
            valid_ladder: "path",
            ladder_up: true,
            ladder_down: true,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 17,
            mountain_max: 1.5,
            water_threshold: 0.7,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.8,
            tree_threshold_perlin: 0.5,
        },
        6 => SmoothConfig {
            default_block: "fire_grass",
            sea_block: "lava",
            coast_block: "sand",
            mountain_block: "stone",
            path_block: "stone",
            inner_mountain_block: "stone",
            ores: FIRE_ORES,
            tree_requirement_block: "fire_grass",
            tree: "fire_tree",
            lava: "lava",
            player_spawn: "fire_grass",
            valid_ladder: "fire_grass",
            ladder_up: true,
            ladder_down: true,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 5,
            mountain_max: 1.0,
            water_threshold: 0.5,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.8,
            tree_threshold_perlin: 0.5,
        },
        7 => SmoothConfig {
            default_block: "ice_grass",
            sea_block: "water",
            coast_block: "ice_grass",
            mountain_block: "stone",
            path_block: "stone",
            inner_mountain_block: "stone",
            ores: ICE_ORES,
            tree_requirement_block: "ice_grass",
            tree: "ice_shrub",
            lava: "water",
            player_spawn: "ice_grass",
            valid_ladder: "ice_grass",
            ladder_up: true,
            ladder_down: true,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 17,
            mountain_max: 1.5,
            water_threshold: 0.5,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.4,
            tree_threshold_perlin: 0.5,
        },
        8 => SmoothConfig {
            default_block: "path",
            sea_block: "path",
            coast_block: "path",
            mountain_block: "wall",
            path_block: "wall",
            inner_mountain_block: "wall",
            ores: BOSS_ORES,
            tree_requirement_block: "path",
            tree: "grave",
            lava: "wall",
            player_spawn: "necromancer",
            valid_ladder: "path",
            ladder_up: false,
            ladder_down: false,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 10,
            mountain_max: 10.0,
            water_threshold: 0.7,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.95,
            tree_threshold_perlin: -1.0,
        },
        _ => SmoothConfig {
            default_block: "grass",
            sea_block: "water",
            coast_block: "sand",
            mountain_block: "stone",
            path_block: "path",
            inner_mountain_block: "path",
            ores: OVERWORLD_ORES,
            tree_requirement_block: "grass",
            tree: "tree",
            lava: "lava",
            player_spawn: "grass",
            valid_ladder: "path",
            ladder_up: false,
            ladder_down: true,
            water_strength: 5,
            water_max: 1.0,
            mountain_strength: 5,
            mountain_max: 1.0,
            water_threshold: 0.7,
            sand_threshold: 0.6,
            tree_threshold_uniform: 0.8,
            tree_threshold_perlin: 0.5,
        },
    }
}

fn dungeon_config(level: i64) -> DungeonConfigRust {
    match level {
        3 => DungeonConfigRust {
            special_block: "enchantment_table_ice",
            fountain_block: "water",
            rare_path_replacement_block: "water",
            valid_ladder: "path",
        },
        4 => DungeonConfigRust {
            special_block: "enchantment_table_fire",
            fountain_block: "fountain",
            rare_path_replacement_block: "path",
            valid_ladder: "path",
        },
        _ => DungeonConfigRust {
            special_block: "path",
            fountain_block: "fountain",
            rare_path_replacement_block: "path",
            valid_ladder: "path",
        },
    }
}

fn empty_item_map(width: i64, height: i64) -> Vec<Vec<String>> {
    vec![vec!["none".to_string(); width as usize]; height as usize]
}

fn fractal_noise_2d(
    rng: &mut PythonRandom,
    width: i64,
    height: i64,
    res: (i64, i64),
) -> Vec<Vec<f64>> {
    let values = perlin_noise_2d(rng, width, height, res);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for row in &values {
        for value in row {
            lo = lo.min(*value);
            hi = hi.max(*value);
        }
    }
    if (lo - hi).abs() <= 1e-9 {
        return vec![vec![0.5; width as usize]; height as usize];
    }
    let scale = hi - lo;
    values
        .into_iter()
        .map(|row| row.into_iter().map(|value| (value - lo) / scale).collect())
        .collect()
}

fn perlin_noise_2d(
    rng: &mut PythonRandom,
    width: i64,
    height: i64,
    res: (i64, i64),
) -> Vec<Vec<f64>> {
    let res_x = std::cmp::max(1, res.0);
    let res_y = std::cmp::max(1, res.1);
    let mut gradients = Vec::new();
    for _ in 0..=res_y {
        let mut row = Vec::new();
        for _ in 0..=res_x {
            let angle = 2.0 * std::f64::consts::PI * rng.random();
            row.push((angle.cos(), angle.sin()));
        }
        gradients.push(row);
    }
    let mut values = Vec::new();
    for y in 0..height {
        let mut row = Vec::new();
        let gy = (y as f64 / std::cmp::max(1, height) as f64) * res_y as f64;
        let iy = std::cmp::min(res_y - 1, gy.floor() as i64);
        let fy = gy - iy as f64;
        for x in 0..width {
            let gx = (x as f64 / std::cmp::max(1, width) as f64) * res_x as f64;
            let ix = std::cmp::min(res_x - 1, gx.floor() as i64);
            let fx = gx - ix as f64;
            let g00 = gradients[iy as usize][ix as usize];
            let g10 = gradients[iy as usize][(ix + 1) as usize];
            let g01 = gradients[(iy + 1) as usize][ix as usize];
            let g11 = gradients[(iy + 1) as usize][(ix + 1) as usize];
            let n00 = dot(g00, fx, fy);
            let n10 = dot(g10, fx - 1.0, fy);
            let n01 = dot(g01, fx, fy - 1.0);
            let n11 = dot(g11, fx - 1.0, fy - 1.0);
            let sx = fade(fx);
            let sy = fade(fy);
            let n0 = lerp(n00, n10, sx);
            let n1 = lerp(n01, n11, sx);
            row.push(2.0_f64.sqrt() * lerp(n0, n1, sy));
        }
        values.push(row);
    }
    values
}

fn choose_ladder(
    grid: &[Vec<String>],
    item_map: &[Vec<String>],
    valid_block: &str,
    rng: &mut PythonRandom,
    avoid: (i64, i64),
) -> (i64, i64) {
    let mut candidates = Vec::new();
    for (y, row) in grid.iter().enumerate() {
        for (x, block) in row.iter().enumerate() {
            let pos = (x as i64, y as i64);
            if pos != avoid && block == valid_block && item_map[y][x] == "none" {
                candidates.push(pos);
            }
        }
    }
    if candidates.is_empty() {
        for (y, row) in grid.iter().enumerate() {
            for (x, block) in row.iter().enumerate() {
                let pos = (x as i64, y as i64);
                if pos != avoid && !matches!(block.as_str(), "wall" | "darkness" | "water" | "lava")
                {
                    candidates.push(pos);
                }
            }
        }
    }
    if candidates.is_empty() {
        let width = grid.first().map(Vec::len).unwrap_or(1) as i64;
        let height = grid.len() as i64;
        return (
            std::cmp::max(0, std::cmp::min(width - 1, 1)),
            std::cmp::max(0, std::cmp::min(height - 1, 1)),
        );
    }
    let index = rng.randrange(0, candidates.len() as i64);
    candidates[index as usize]
}

/// A `densities` entry scales how much of a feature the generator produces,
/// relative to the vanilla amount.  `1.0` is vanilla, `0.0` removes the
/// feature, `0.5` is half as much.  It is never an absolute fraction of the
/// map.
fn density(densities: Option<&Value>, key: &str, default: f64) -> f64 {
    densities
        .and_then(|value| value.get(key))
        .and_then(Value::as_f64)
        .unwrap_or(default)
        .max(0.0)
}

fn tree_uniform_threshold(threshold: f64, density: f64) -> f64 {
    let chance = (1.0 - threshold).max(0.0) * density.max(0.0);
    (1.0 - chance).clamp(0.0, 1.0)
}

/// Scale the sea and coast cut points instead of distorting the water field.
///
/// Compressing the noise toward `water_threshold` (the previous behaviour)
/// pulls every tile into the narrow band between `sand_threshold` and
/// `water_threshold`, so a low density turned the whole surface into coast and
/// erased the default block.  Trees require the default block, so `water: 0.05`
/// silently produced a treeless world on every seed.  Moving the cuts upward
/// shrinks sea and coast together and leaves the default block in place.
fn water_cuts(water_threshold: f64, sand_threshold: f64, density: f64) -> (f64, f64) {
    if (density - 1.0).abs() <= f64::EPSILON {
        return (water_threshold, sand_threshold);
    }
    let shrink = |cut: f64| -> f64 {
        if density >= 1.0 {
            // More water: move the cut down toward the field minimum.
            cut - (density - 1.0) * (cut + 1.0)
        } else {
            // Less water: move the cut up toward the field maximum.
            cut + (1.0 - density) * (1.0 - cut)
        }
    };
    (
        shrink(water_threshold).clamp(-1.0, 1.0),
        shrink(sand_threshold).clamp(-1.0, 1.0),
    )
}

fn dot(gradient: (f64, f64), x: f64, y: f64) -> f64 {
    gradient.0 * x + gradient.1 * y
}

fn fade(value: f64) -> f64 {
    value * value * value * (value * (value * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f64, b: f64, weight: f64) -> f64 {
    a * (1.0 - weight) + b * weight
}

fn floor_mobs(level: i64) -> &'static [&'static str] {
    match level {
        1 => &["snail", "orc_solider", "orc_mage"],
        2 => &["bat", "gnome_warrior", "gnome_archer"],
        3 => &["snail", "lizard", "kobold"],
        4 => &["snail", "knight", "archer"],
        5 => &["bat", "troll", "deep_thing"],
        6 => &["bat", "pigman", "fire_elemental"],
        7 => &["bat", "frost_troll", "ice_elemental"],
        _ => &["cow", "zombie", "skeleton"],
    }
}

fn mob_health(kind: &str) -> i64 {
    match kind {
        "cow" | "skeleton" => 3,
        "bat" | "deep_thing" => 4,
        "zombie" | "gnome_archer" => 5,
        "snail" | "orc_mage" => 6,
        "gnome_warrior" => 7,
        "kobold" | "necromancer" => 8,
        "orc_solider" => 9,
        "lizard" => 11,
        "knight" | "archer" => 12,
        "fire_elemental" => 14,
        "ice_elemental" => 16,
        "troll" | "pigman" => 20,
        "frost_troll" => 24,
        _ => 3,
    }
}

fn spawn_mob_kind(level: i64, class_name: &str) -> &'static str {
    let choices = floor_mobs(level);
    match class_name {
        "passive" => choices[0],
        "melee" => choices[1],
        "ranged" => choices[2],
        _ => choices[0],
    }
}

fn floor_spawn_chance(level: i64, class_index: usize) -> f64 {
    const TABLE: [[f64; 4]; 9] = [
        [0.1, 0.02, 0.05, 0.1],
        [0.1, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
        [0.0, 0.06, 0.05, 0.0],
        [0.1, 0.06, 0.05, 0.0],
    ];
    TABLE[level.clamp(0, 8) as usize][class_index]
}

fn melee_damage_vector(kind: &str) -> [f64; 3] {
    match mob_type_id(kind) {
        0 => [2.0, 0.0, 0.0],
        1 => [4.0, 0.0, 0.0],
        2 => [3.0, 0.0, 0.0],
        3 => [5.0, 0.0, 0.0],
        4 => [6.0, 0.0, 0.0],
        5 => [6.0, 1.0, 1.0],
        6 => [3.0, 5.0, 0.0],
        7 => [4.0, 0.0, 5.0],
        _ => [2.0, 0.0, 0.0],
    }
}

fn projectile_damage(kind: &str) -> f64 {
    match kind {
        "fireball" | "iceball" => 3.0,
        "arrow" | "arrow2" => 2.0,
        _ => 1.0,
    }
}

fn mob_achievement(kind: &str) -> Option<&'static str> {
    match kind {
        "zombie" => Some("defeat_zombie"),
        "skeleton" => Some("defeat_skeleton"),
        "gnome_warrior" => Some("defeat_gnome_warrior"),
        "gnome_archer" => Some("defeat_gnome_archer"),
        "orc_solider" => Some("defeat_orc_solider"),
        "orc_mage" => Some("defeat_orc_mage"),
        "lizard" => Some("defeat_lizard"),
        "kobold" => Some("defeat_kobold"),
        "knight" => Some("defeat_knight"),
        "archer" => Some("defeat_archer"),
        "troll" => Some("defeat_troll"),
        "deep_thing" => Some("defeat_deep_thing"),
        "pigman" => Some("defeat_pigman"),
        "fire_elemental" => Some("defeat_fire_elemental"),
        "frost_troll" => Some("defeat_frost_troll"),
        "ice_elemental" => Some("defeat_ice_elemental"),
        "necromancer" | "necromancer_vulnerable" => Some("defeat_necromancer"),
        _ => None,
    }
}

fn passive_mob_achievement(kind: &str) -> Option<&'static str> {
    match kind {
        "cow" => Some("eat_cow"),
        "bat" => Some("eat_bat"),
        "snail" => Some("eat_snail"),
        _ => None,
    }
}

fn level_achievement(level: i64) -> Option<&'static str> {
    match level {
        1 => Some("enter_dungeon"),
        2 => Some("enter_gnomish_mines"),
        3 => Some("enter_sewers"),
        4 => Some("enter_vault"),
        5 => Some("enter_troll_mines"),
        6 => Some("enter_fire_realm"),
        7 => Some("enter_ice_realm"),
        8 => Some("enter_graveyard"),
        _ => None,
    }
}

fn calculate_light_level(timestep: i64, day_length: i64) -> f64 {
    let progress = (timestep as f64 / day_length.max(1) as f64) % 1.0 + 0.3;
    1.0 - (std::f64::consts::PI * progress).cos().abs().powi(3)
}

fn day_length_from_resolved(resolved: &ResolvedTask) -> i64 {
    resolved
        .world
        .get("day_length")
        .or_else(|| resolved.rules.get("day_length"))
        .and_then(Value::as_i64)
        .unwrap_or(300)
        .max(1)
}

fn add_pos(left: (i64, i64), right: (i64, i64)) -> (i64, i64) {
    (left.0 + right.0, left.1 + right.1)
}

fn manhattan(left: (i64, i64), right: (i64, i64)) -> i64 {
    (left.0 - right.0).abs() + (left.1 - right.1).abs()
}

fn euclidean(left: (i64, i64), right: (i64, i64)) -> f64 {
    (((left.0 - right.0).pow(2) + (left.1 - right.1).pow(2)) as f64).sqrt()
}

fn sign(value: i64) -> i64 {
    if value > 0 {
        1
    } else if value < 0 {
        -1
    } else {
        0
    }
}

fn mob_class(kind: &str) -> &'static str {
    match kind {
        "cow" | "bat" | "snail" => "passive",
        "skeleton" | "gnome_archer" | "orc_mage" | "kobold" | "archer" | "deep_thing"
        | "fire_elemental" | "ice_elemental" => "ranged",
        _ => "melee",
    }
}

fn mob_type_id(kind: &str) -> usize {
    match kind {
        "bat" | "gnome_warrior" | "gnome_archer" => 1,
        "snail" | "orc_solider" | "orc_mage" => 2,
        "lizard" | "kobold" => 3,
        "knight" | "archer" => 4,
        "troll" | "deep_thing" => 5,
        "pigman" | "fire_elemental" => 6,
        "frost_troll" | "ice_elemental" => 7,
        _ => 0,
    }
}

fn mob_class_index(mob_class: &str) -> usize {
    match mob_class {
        "passive" => 0,
        "melee" => 1,
        "ranged" => 2,
        "projectile" => 3,
        _ => 1,
    }
}

fn mob_can_occupy_block(kind: &str, mob_class: &str, block: &str) -> bool {
    if static_solid(block) {
        return false;
    }
    let collision = collision_map(mob_type_id(kind), mob_class_index(mob_class));
    let in_water = block == "water";
    let in_lava = block == "lava";
    let on_ground = !static_solid(block) && !in_water && !in_lava;
    if collision[0] && on_ground {
        return false;
    }
    if collision[1] && in_water {
        return false;
    }
    if collision[2] && in_lava {
        return false;
    }
    true
}

fn collision_map(mob_type: usize, class_index: usize) -> [bool; 3] {
    const LAND: [bool; 3] = [false, true, true];
    const FLYING: [bool; 3] = [false, false, false];
    const AQUATIC: [bool; 3] = [true, false, true];
    const AMPHIBIAN: [bool; 3] = [false, false, true];
    const TABLE: [[[bool; 3]; 4]; 9] = [
        [LAND, LAND, LAND, FLYING],
        [FLYING, LAND, LAND, FLYING],
        [LAND, LAND, LAND, FLYING],
        [LAND, AMPHIBIAN, LAND, FLYING],
        [LAND, LAND, LAND, FLYING],
        [LAND, LAND, AQUATIC, FLYING],
        [LAND, LAND, FLYING, FLYING],
        [LAND, LAND, FLYING, FLYING],
        [LAND, LAND, LAND, FLYING],
    ];
    TABLE[mob_type][class_index]
}

fn static_solid(block: &str) -> bool {
    SOLID_BLOCKS.contains(&block)
        && !matches!(
            block,
            "water" | "lava" | "darkness" | "ice_shrub" | "necromancer_vulnerable"
        )
}

fn normalize_tile(kind: &str) -> &str {
    match kind {
        "table" => "crafting_table",
        "torch" => "torch",
        other => other,
    }
}

fn normalize_item(kind: &str) -> &str {
    match kind {
        "torch" | "ladder_down" | "ladder_up" | "ladder_down_blocked" => kind,
        _ => "none",
    }
}

fn item_overlay(kind: &str) -> bool {
    matches!(
        kind,
        "torch" | "ladder_down" | "ladder_up" | "ladder_down_blocked"
    )
}

fn set_item_cell(world: &mut CraftaxWorld, level: i64, pos: (i64, i64), kind: &str) {
    let item = normalize_item(kind);
    let level_index = level as usize;
    if item == "ladder_down" || item == "ladder_up" {
        for row in &mut world.item_maps[level_index] {
            for existing in row {
                if existing == item {
                    *existing = "none".to_string();
                }
            }
        }
    }
    world.item_maps[level_index][pos.1 as usize][pos.0 as usize] = item.to_string();
    if item == "ladder_down" && level_index < world.down_ladders.len() {
        world.down_ladders[level_index] = pos;
    } else if item == "ladder_up" && level_index < world.up_ladders.len() {
        world.up_ladders[level_index] = pos;
    }
}

fn find_item(map: &[Vec<String>], item: &str) -> Option<(i64, i64)> {
    for (y, row) in map.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            if value == item {
                return Some((x as i64, y as i64));
            }
        }
    }
    None
}

fn valid_ladder_pos(pos: Option<(i64, i64)>) -> Option<(i64, i64)> {
    pos.filter(|(x, y)| *x >= 0 && *y >= 0)
}

fn normalize_mob(kind: &str) -> &str {
    match kind {
        "goblin" => "gnome_warrior",
        "orc_soldier" => "orc_solider",
        "knight_archer" => "archer",
        other => other,
    }
}

fn normalize_action_value(value: &Value) -> Result<String> {
    let action = value
        .as_str()
        .ok_or_else(|| anyhow!("action must be string for current Rust lane: {value:?}"))?;
    Ok(match action {
        "move_left" => "left",
        "move_right" => "right",
        "move_up" => "up",
        "move_down" => "down",
        "wait" => "noop",
        "cast_fireball" | "cast_iceball" => "cast_spell",
        other => other,
    }
    .to_string())
}

fn dir_for_action(action: &str) -> Option<(i64, i64)> {
    match action {
        "left" => Some((-1, 0)),
        "right" => Some((1, 0)),
        "up" => Some((0, -1)),
        "down" => Some((0, 1)),
        _ => None,
    }
}

fn is_land_walkable(tile: &str) -> bool {
    !static_solid(tile) && tile != "water" && tile != "lava"
}

fn base_walkable(tile: &str) -> bool {
    matches!(
        tile,
        "grass" | "path" | "sand" | "fire_grass" | "ice_grass" | "gravel"
    )
}

fn can_place_item_on(tile: &str) -> bool {
    matches!(tile, "grass" | "sand" | "path" | "fire_grass" | "ice_grass")
}

fn resource_tile(tile: &str) -> Option<(&'static str, &'static str, i64)> {
    match tile {
        "tree" | "fire_tree" | "ice_shrub" => Some(("wood", "collect_wood", 0)),
        "stone" | "stalagmite" => Some(("stone", "collect_stone", 1)),
        "coal" => Some(("coal", "collect_coal", 1)),
        "iron" => Some(("iron", "collect_iron", 2)),
        "diamond" => Some(("diamond", "collect_diamond", 3)),
        "sapphire" => Some(("sapphire", "collect_sapphire", 4)),
        "ruby" => Some(("ruby", "collect_ruby", 4)),
        _ => None,
    }
}

fn resource_replacement(tile: &str, level: i64) -> &'static str {
    match tile {
        "tree" => "grass",
        "fire_tree" => "fire_grass",
        "ice_shrub" => "ice_grass",
        _ if level > 0 => "path",
        _ => "grass",
    }
}

fn tier_name(tier: i64) -> &'static str {
    match tier {
        1 => "wood",
        2 => "stone",
        3 => "iron",
        4 => "diamond",
        _ => "none",
    }
}

struct Recipe {
    costs: &'static [(&'static str, i64)],
    target: &'static str,
    tier: i64,
    amount: i64,
}

const COST_WOOD_1: &[(&str, i64)] = &[("wood", 1)];
const COST_WOOD_STONE: &[(&str, i64)] = &[("wood", 1), ("stone", 1)];
const COST_IRON_TOOL: &[(&str, i64)] = &[("wood", 1), ("stone", 1), ("iron", 1), ("coal", 1)];
const COST_DIAMOND_PICKAXE: &[(&str, i64)] = &[("wood", 1), ("diamond", 3)];
const COST_DIAMOND_SWORD: &[(&str, i64)] = &[("wood", 1), ("diamond", 2)];
const COST_TORCH: &[(&str, i64)] = &[("wood", 1), ("coal", 1)];
const COST_IRON_ARMOUR: &[(&str, i64)] = &[("iron", 3), ("coal", 3)];
const COST_DIAMOND_ARMOUR: &[(&str, i64)] = &[("diamond", 3)];

fn recipe_for(action: &str) -> Option<Recipe> {
    Some(match action {
        "make_wood_pickaxe" => Recipe {
            costs: COST_WOOD_1,
            target: "pickaxe",
            tier: 1,
            amount: 1,
        },
        "make_stone_pickaxe" => Recipe {
            costs: COST_WOOD_STONE,
            target: "pickaxe",
            tier: 2,
            amount: 1,
        },
        "make_iron_pickaxe" => Recipe {
            costs: COST_IRON_TOOL,
            target: "pickaxe",
            tier: 3,
            amount: 1,
        },
        "make_diamond_pickaxe" => Recipe {
            costs: COST_DIAMOND_PICKAXE,
            target: "pickaxe",
            tier: 4,
            amount: 1,
        },
        "make_wood_sword" => Recipe {
            costs: COST_WOOD_1,
            target: "sword",
            tier: 1,
            amount: 1,
        },
        "make_stone_sword" => Recipe {
            costs: COST_WOOD_STONE,
            target: "sword",
            tier: 2,
            amount: 1,
        },
        "make_iron_sword" => Recipe {
            costs: COST_IRON_TOOL,
            target: "sword",
            tier: 3,
            amount: 1,
        },
        "make_diamond_sword" => Recipe {
            costs: COST_DIAMOND_SWORD,
            target: "sword",
            tier: 4,
            amount: 1,
        },
        "make_arrow" => Recipe {
            costs: COST_WOOD_STONE,
            target: "arrows",
            tier: 0,
            amount: 2,
        },
        "make_torch" => Recipe {
            costs: COST_TORCH,
            target: "torches",
            tier: 0,
            amount: 4,
        },
        "make_iron_armour" => Recipe {
            costs: COST_IRON_ARMOUR,
            target: "armour",
            tier: 1,
            amount: 1,
        },
        "make_diamond_armour" => Recipe {
            costs: COST_DIAMOND_ARMOUR,
            target: "armour",
            tier: 2,
            amount: 1,
        },
        _ => return None,
    })
}

fn costs_json(costs: &[(&str, i64)]) -> Value {
    let mut object = Map::new();
    for (key, value) in costs {
        object.insert((*key).to_string(), json!(value));
    }
    Value::Object(object)
}

fn achievement_reward(_achievement: &str) -> f64 {
    1.0
}

fn max_stat(world: &CraftaxWorld, stat: &str) -> Result<i64> {
    let inv = as_object(&world.inventory, "inventory")?;
    let value = match stat {
        "health" => 8 + inv.get("strength").and_then(Value::as_i64).unwrap_or(1),
        "food" | "drink" | "energy" => {
            7 + 2 * inv.get("dexterity").and_then(Value::as_i64).unwrap_or(1)
        }
        "mana" => 6 + 3 * inv.get("intelligence").and_then(Value::as_i64).unwrap_or(1),
        _ => 9,
    };
    Ok(value)
}

fn potion_index(color: &str) -> Result<usize> {
    match color {
        "red" => Ok(0),
        "green" => Ok(1),
        "blue" => Ok(2),
        "pink" => Ok(3),
        "cyan" => Ok(4),
        "yellow" => Ok(5),
        _ => bail!("unknown potion color: {color}"),
    }
}

fn armour_vec(world: &CraftaxWorld) -> Result<Vec<i64>> {
    let Some(value) = world.inventory.get("armour") else {
        return Ok(vec![0, 0, 0, 0]);
    };
    if let Some(number) = value.as_i64() {
        return Ok(vec![number; 4]);
    }
    let mut pieces = value
        .as_array()
        .ok_or_else(|| anyhow!("inventory.armour must be int or array"))?
        .iter()
        .take(4)
        .map(|value| strict_int(value, "inventory.armour[]"))
        .collect::<Result<Vec<_>>>()?;
    pieces.resize(4, 0);
    Ok(pieces)
}

/// One glyph per world feature, shared by `local_map`, `ascii_map`, and the
/// legend handed to policies.
///
/// The table is the single source of truth and is asserted injective by
/// `glyph_table_is_injective`.  Ambiguity here is not cosmetic: entity glyphs
/// used to be `kind[0].to_ascii_uppercase()`, which rendered a pigman as `P`
/// (the player), a troll as `T` (a tree), and a lizard as `L` (lava).
pub const TILE_GLYPHS: &[(&str, char, &str)] = &[
    ("grass", '.', "grass, walkable; `do` may yield a sapling"),
    ("path", '_', "path, walkable and inert"),
    ("sand", ',', "sand, walkable and inert"),
    ("gravel", ';', "gravel, walkable and inert"),
    ("fire_grass", '"', "fire grass, walkable"),
    ("ice_grass", '\'', "ice grass, walkable"),
    ("water", '~', "water, blocks movement; `do` drinks"),
    ("fountain", 'u', "fountain, blocks movement; `do` drinks"),
    ("stone", 'o', "stone, blocks movement; `do` mines it with a wood pickaxe"),
    ("stalagmite", '^', "stalagmite, mines like stone"),
    ("wall", '#', "wall"),
    ("wall_moss", '%', "mossy wall"),
    ("tree", 'T', "tree; `do` yields wood, no tool needed"),
    ("fire_tree", 'Y', "fire tree; yields wood"),
    ("ice_shrub", 'y', "ice shrub; yields wood"),
    ("lava", 'L', "lava, lethal"),
    ("coal", 'c', "coal; needs a wood pickaxe"),
    ("iron", 'i', "iron; needs a stone pickaxe"),
    ("diamond", 'd', "diamond; needs an iron pickaxe"),
    ("sapphire", 's', "sapphire; needs a diamond pickaxe"),
    ("ruby", 'r', "ruby; needs a diamond pickaxe"),
    ("chest", 'h', "chest; `do` opens it"),
    ("crafting_table", 'a', "crafting table"),
    ("furnace", 'F', "furnace"),
    ("ladder_down", '>', "ladder down; stand on it and `descend`"),
    ("ladder_up", '<', "ladder up; stand on it and `ascend`"),
    ("ladder_down_blocked", 'x', "down ladder, not yet open"),
    ("plant", 'p', "planted sapling"),
    ("ripe_plant", 'R', "ripe plant; `do` eats it"),
    ("torch", 't', "torch"),
    ("grave", '+', "grave"),
    ("grave2", '+', "grave"),
    ("grave3", '+', "grave"),
    ("enchantment_table_fire", 'E', "fire enchantment table"),
    ("enchantment_table_ice", 'e', "ice enchantment table"),
    ("necromancer", 'N', "necromancer, not vulnerable"),
    ("necromancer_vulnerable", 'n', "necromancer, vulnerable"),
    ("darkness", ' ', "unseen"),
    ("out_of_bounds", ' ', "outside the world"),
    ("invalid", '?', "unknown"),
];

/// Mob glyphs.  Distinct from every tile glyph and from the player.
pub const ENTITY_GLYPHS: &[(&str, char)] = &[
    ("cow", 'C'),
    ("zombie", 'Z'),
    ("skeleton", 'S'),
    ("bat", 'B'),
    ("snail", 'U'),
    ("orc_solider", 'O'),
    ("orc_mage", 'Q'),
    ("gnome_warrior", 'G'),
    ("gnome_archer", 'g'),
    ("lizard", 'V'),
    ("kobold", 'K'),
    ("knight", 'W'),
    ("archer", 'A'),
    ("troll", 'M'),
    ("deep_thing", 'D'),
    ("pigman", 'X'),
    ("fire_elemental", 'H'),
    ("frost_troll", 'I'),
    ("ice_elemental", 'J'),
    ("necromancer", 'N'),
];

/// The fixed action vocabulary, published once via `/info`.
pub fn action_names() -> Vec<&'static str> {
    ACTION_NAMES.to_vec()
}

pub const PLAYER_GLYPH: char = 'P';
pub const UNKNOWN_GLYPH: char = '?';

fn tile_char(tile: &str) -> char {
    TILE_GLYPHS
        .iter()
        .find(|(name, _, _)| *name == tile)
        .map(|(_, glyph, _)| *glyph)
        .unwrap_or(UNKNOWN_GLYPH)
}

fn entity_char(kind: &str) -> char {
    ENTITY_GLYPHS
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, glyph)| *glyph)
        .unwrap_or(UNKNOWN_GLYPH)
}

fn projectile_char(kind: &str, owner: &str) -> char {
    if owner == "mob" {
        '!'
    } else {
        match kind {
            "fireball" | "fireball2" => '*',
            "iceball" | "iceball2" => ':',
            _ => '-',
        }
    }
}

/// Render the legend that policies receive, straight from the tables above so
/// prompt text cannot drift from what the renderer emits.
pub fn glyph_legend() -> String {
    let mut lines = vec![
        "Map glyphs, each denoting exactly one thing:".to_string(),
        format!("  {PLAYER_GLYPH}  you"),
    ];
    let mut seen = HashSet::new();
    for (_, glyph, description) in TILE_GLYPHS {
        if *glyph == ' ' || !seen.insert(*glyph) {
            continue;
        }
        lines.push(format!("  {glyph}  {description}"));
    }
    lines.push("  (space)  unseen, or outside the world".to_string());
    let mobs = ENTITY_GLYPHS
        .iter()
        .filter(|(name, _)| *name != "necromancer")
        .map(|(name, glyph)| format!("{glyph} {}", name.replace('_', " ")))
        .collect::<Vec<_>>()
        .join(", ");
    lines.push(format!("Mobs: {mobs}."));
    lines.push(
        "Projectiles: ! incoming mob projectile, * your fireball, : your iceball, \
         - your arrow."
            .to_string(),
    );
    lines.join("\n")
}

/// True when an inventory slot is at its starting value and carries no
/// information.  Stats that start full (health, food, drink, energy, mana) are
/// always shown, because their decline is the signal.
fn is_default_inventory(key: &str, value: &Value) -> bool {
    if matches!(
        key,
        "health" | "food" | "drink" | "energy" | "mana" | "xp"
    ) {
        return false;
    }
    match key {
        "dexterity" | "strength" | "intelligence" => value.as_i64() == Some(1),
        "armour" => value
            .as_array()
            .is_some_and(|slots| slots.iter().all(|slot| slot.as_i64() == Some(0))),
        "armour_enchantments" => value
            .as_array()
            .is_some_and(|slots| slots.iter().all(|slot| slot.as_str() == Some("none"))),
        "sword_enchantment" | "bow_enchantment" => value.as_str() == Some("none"),
        _ => value.as_i64() == Some(0) || value.as_f64() == Some(0.0),
    }
}

fn observation_text(observation: &Value) -> String {
    let inventory = observation
        .get("inventory")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let inventory_object = inventory.as_object().cloned().unwrap_or_default();
    // Only carry what is not at its default.  A twenty-turn ReAct history
    // repeats this line every turn, and spelling out thirty zero-valued slots
    // each time crowded out the part of the context that changes.
    let mut inventory_parts = Vec::new();
    for (key, value) in inventory_object {
        if key == "potions" || key == "learned_spells" || is_default_inventory(&key, &value) {
            continue;
        }
        inventory_parts.push(format!("{key}={value}"));
    }
    let potions = observation
        .pointer("/inventory/potions")
        .and_then(Value::as_object)
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let learned_spells = observation
        .pointer("/inventory/learned_spells")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let local_map = observation
        .get("local_map")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let achievements = observation
        .get("achievements")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let nearby_entities = observation
        .get("nearby_entities")
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .map(|entity| {
                    format!(
                        "{}@{} hp={}",
                        entity.get("kind").and_then(Value::as_str).unwrap_or("?"),
                        entity.get("pos").cloned().unwrap_or(Value::Null),
                        entity.get("health").cloned().unwrap_or(Value::Null)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let projectiles = observation
        .get("projectile_state")
        .and_then(Value::as_object)
        .map(|groups| {
            groups
                .values()
                .flat_map(|value| value.as_array().cloned().unwrap_or_default())
                .map(|projectile| {
                    format!(
                        "{}:{}@{} dir={}",
                        projectile
                            .get("owner")
                            .and_then(Value::as_str)
                            .unwrap_or("?"),
                        projectile
                            .get("kind")
                            .and_then(Value::as_str)
                            .unwrap_or("?"),
                        projectile.get("pos").cloned().unwrap_or(Value::Null),
                        projectile.get("direction").cloned().unwrap_or(Value::Null)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    let mut lines = vec![
        format!(
            "level: {}",
            observation
                .pointer("/player/level")
                .cloned()
                .unwrap_or(Value::Null)
        ),
        format!(
            "position: {} direction={}",
            observation
                .pointer("/player/pos")
                .cloned()
                .unwrap_or(Value::Null),
            observation
                .pointer("/player/direction")
                .cloned()
                .unwrap_or(Value::Null)
        ),
        format!(
            "front_tile: {}",
            observation
                .pointer("/player/front_tile")
                .and_then(Value::as_str)
                .unwrap_or("out_of_bounds")
        ),
        "local_map:".to_string(),
    ];
    lines.extend(local_map);
    lines.push(format!(
        "inventory: {} (anything not listed is 0)",
        if inventory_parts.is_empty() {
            "empty".to_string()
        } else {
            inventory_parts.join(", ")
        }
    ));
    for (label, value) in [
        ("potions", &potions),
        ("learned_spells", &learned_spells),
        ("achievements", &achievements),
        ("nearby_entities", &nearby_entities),
        ("projectiles", &projectiles),
    ] {
        if !value.is_empty() {
            lines.push(format!("{label}: {value}"));
        }
    }
    // The action vocabulary is fixed and is published once via `/info` and the
    // tool schema.  Repeating it per turn cost ~90 tokens a turn and grew
    // quadratically as the conversation accumulated.
    lines.join("\n")
}

fn pos2(value: &Value, field: &str) -> Result<(i64, i64)> {
    let array = value
        .as_array()
        .ok_or_else(|| anyhow!("{field} must be a two-element coordinate: {value:?}"))?;
    if array.len() != 2 {
        bail!("{field} must be a two-element coordinate: {value:?}");
    }
    Ok((
        strict_int(&array[0], &format!("{field}[0]"))?,
        strict_int(&array[1], &format!("{field}[1]"))?,
    ))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PythonRandomState {
    mt: Vec<u32>,
    index: usize,
}

#[derive(Debug, Clone)]
struct PythonRandom {
    mt: [u32; MT_N],
    index: usize,
}

impl PythonRandom {
    fn to_state(&self) -> PythonRandomState {
        PythonRandomState {
            mt: self.mt.to_vec(),
            index: self.index,
        }
    }

    fn from_state(state: PythonRandomState) -> Result<Self> {
        if state.mt.len() != MT_N {
            bail!("invalid RNG state length: {}", state.mt.len());
        }
        if state.index > MT_N {
            bail!("invalid RNG state index: {}", state.index);
        }
        let mut mt = [0; MT_N];
        mt.copy_from_slice(&state.mt);
        Ok(Self {
            mt,
            index: state.index,
        })
    }

    fn seed_from_i64(seed: i64) -> Self {
        let mut rng = Self {
            mt: [0; MT_N],
            index: MT_N,
        };
        let abs_seed = if seed < 0 {
            seed.unsigned_abs()
        } else {
            seed as u64
        };
        let key = if abs_seed <= u32::MAX as u64 {
            vec![abs_seed as u32]
        } else {
            vec![(abs_seed & 0xffff_ffff) as u32, (abs_seed >> 32) as u32]
        };
        rng.init_by_array(&key);
        rng
    }

    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..MT_N {
            self.mt[i] = 1812433253u32
                .wrapping_mul(self.mt[i - 1] ^ (self.mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        self.index = MT_N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = MT_N.max(key.len());
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1664525)))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = MT_N - 1;
        while k > 0 {
            self.mt[i] = (self.mt[i]
                ^ ((self.mt[i - 1] ^ (self.mt[i - 1] >> 30)).wrapping_mul(1566083941)))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= MT_N {
                self.mt[0] = self.mt[MT_N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    fn gen_u32(&mut self) -> u32 {
        if self.index >= MT_N {
            self.twist();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn twist(&mut self) {
        for kk in 0..(MT_N - MT_M) {
            let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
            self.mt[kk] = self.mt[kk + MT_M] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        }
        for kk in (MT_N - MT_M)..(MT_N - 1) {
            let y = (self.mt[kk] & MT_UPPER_MASK) | (self.mt[kk + 1] & MT_LOWER_MASK);
            self.mt[kk] =
                self.mt[kk + MT_M - MT_N] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        }
        let y = (self.mt[MT_N - 1] & MT_UPPER_MASK) | (self.mt[0] & MT_LOWER_MASK);
        self.mt[MT_N - 1] = self.mt[MT_M - 1] ^ (y >> 1) ^ if y & 1 != 0 { MT_MATRIX_A } else { 0 };
        self.index = 0;
    }

    fn random(&mut self) -> f64 {
        let a = (self.gen_u32() >> 5) as u64;
        let b = (self.gen_u32() >> 6) as u64;
        ((a << 26) + b) as f64 / 9007199254740992.0
    }

    fn getrandbits(&mut self, bits: u32) -> u64 {
        if bits == 0 {
            return 0;
        }
        let words = bits.div_ceil(32);
        let mut value = 0u64;
        for word in 0..words {
            value |= (self.gen_u32() as u64) << (32 * word);
        }
        let excess = words * 32 - bits;
        if excess > 0 {
            value >>= excess;
        }
        value
    }

    fn randbelow(&mut self, n: i64) -> i64 {
        let bits = (i64::BITS - n.leading_zeros()) as u32;
        loop {
            let r = self.getrandbits(bits) as i64;
            if r < n {
                return r;
            }
        }
    }

    fn randrange(&mut self, start: i64, stop: i64) -> i64 {
        start + self.randbelow(stop - start)
    }

    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.randbelow((i + 1) as i64) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn survival_session() -> CraftaxRustSession {
        CraftaxRustSession::reset_from_task(
            &json!({
                "schema": "gamebench.task.craftax.v1",
                "task_id": "intrinsics-test",
                "scenario_id": "intrinsics-test",
                "seed": 0,
                "world": {"use_default": "policy_dev_small", "max_steps": 1000},
                "rules": {"base": "symbolic_survival"},
                "readouts": {"profile": "symbolic_compact"}
            }),
            None,
        )
        .expect("reset survival session")
    }

    #[test]
    fn jax_intrinsic_thresholds_deplete_food_drink_and_energy() {
        let mut session = survival_session();
        for _ in 0..26 {
            session.update_intrinsics("noop").expect("intrinsic tick");
        }
        assert_eq!(session.inventory_f64("food").unwrap(), 8.0);
        assert_eq!(session.inventory_f64("drink").unwrap(), 8.0);
        assert_eq!(session.inventory_f64("energy").unwrap(), 9.0);
        assert_eq!(session.world.player_hunger, 0.0);
        assert_eq!(session.world.player_thirst, 5.0);
        assert_eq!(session.world.player_fatigue, 26.0);

        for _ in 0..5 {
            session.update_intrinsics("noop").expect("intrinsic tick");
        }
        assert_eq!(session.inventory_f64("energy").unwrap(), 8.0);
        assert_eq!(session.world.player_fatigue, 0.0);
    }

    #[test]
    fn depleted_necessities_cause_terminal_death() {
        let mut session = survival_session();
        session.set_inventory_number("health", 1.0).unwrap();
        session.set_inventory_number("food", 0.0).unwrap();
        session.set_inventory_number("drink", 0.0).unwrap();
        session.set_inventory_number("energy", 0.0).unwrap();
        session.world.player_recover = -15.0;

        session.step(&json!("noop")).expect("fatal step");

        assert!(session.is_done());
        assert_eq!(session.inventory_f64("health").unwrap(), 0.0);
        assert_eq!(session.private["done_reason"], json!("death"));
        assert_eq!(session.private["terminated"], json!(true));
        assert_eq!(session.private["truncated"], json!(false));
    }

    #[test]
    fn every_achievement_has_unit_reward() {
        for achievement in [
            "collect_wood",
            "enter_dungeon",
            "open_chest",
            "enter_fire_realm",
            "defeat_necromancer",
        ] {
            assert_eq!(achievement_reward(achievement), 1.0);
        }
    }

    #[test]
    fn world_advancing_combat_actions_replay_from_action_events() {
        let task = json!({
            "schema": "gamebench.task.craftax.v1",
            "task_id": "action-applied-replay",
            "scenario_id": "action-applied-replay",
            "seed": 0,
            "world": {
                "use_default": "fixture_room",
                "seed": 0,
                "max_passive_mobs": 0,
                "max_melee_mobs": 0,
                "max_ranged_mobs": 0,
                "initial_state": {
                    "player": {"pos": [4, 4], "direction": [1, 0], "level": 0},
                    "inventory": {
                        "sword": 1,
                        "mana": 2,
                        "bow": 1,
                        "arrows": 1,
                        "learned_spells": ["fireball"]
                    },
                    "entities": [
                        {"kind": "zombie", "pos": [5, 4], "level": 0, "health": 5}
                    ]
                }
            },
            "rules": {"base": "symbolic_no_homeostasis"}
        });
        let submitted_actions = vec![json!("do"), json!("shoot_arrow"), json!("cast_spell")];
        let mut original = CraftaxRustSession::reset_from_task(&task, None).unwrap();
        for action in &submitted_actions {
            original.step(action).unwrap();
        }
        let replay_tape: Vec<Value> = original
            .events
            .iter()
            .filter(|event| event.kind == "action_applied")
            .filter_map(|event| event.action.clone())
            .collect();
        assert_eq!(replay_tape, submitted_actions);

        let original_readout = original.readout();
        let mut replay = CraftaxRustSession::reset_from_task(&task, None).unwrap();
        for action in &replay_tape {
            replay.step(action).unwrap();
        }
        assert_eq!(replay.readout(), original_readout);
    }

    #[test]
    fn placement_cannot_create_a_solid_block_under_an_entity() {
        let task = json!({
            "schema": "gamebench.task.craftax.v1",
            "task_id": "occupied-placement",
            "scenario_id": "occupied-placement",
            "seed": 0,
            "world": {
                "use_default": "fixture_room",
                "seed": 0,
                "max_passive_mobs": 0,
                "max_melee_mobs": 0,
                "max_ranged_mobs": 0,
                "initial_state": {
                    "player": {"pos": [4, 4], "direction": [1, 0], "level": 0},
                    "inventory": {"wood": 2},
                    "entities": [
                        {"kind": "zombie", "pos": [5, 4], "level": 0, "health": 5}
                    ]
                }
            },
            "rules": {"base": "symbolic_no_homeostasis"}
        });
        let mut session = CraftaxRustSession::reset_from_task(&task, None).unwrap();
        session.set_tile((5, 4), "grass");

        session.apply_place("place_table").unwrap();

        assert_eq!(session.block_at((5, 4)), "grass");
        assert_eq!(session.inventory_i64("wood").unwrap(), 2);
        let event = session.events.last().expect("blocked placement event");
        assert_eq!(event.transition.as_deref(), Some("blocked_by_mob"));
        assert_eq!(event.payload["target"], json!([5, 4]));
        assert_eq!(event.payload["entity"]["kind"], json!("zombie"));
    }
}

#[cfg(test)]
mod glyph_tests {
    use super::*;
    use std::collections::HashMap;

    /// Every glyph must denote exactly one thing.  Before this table existed,
    /// mob glyphs were `kind[0].to_ascii_uppercase()`: pigman rendered as `P`
    /// (indistinguishable from the player), troll as `T` (a tree), lizard as
    /// `L` (lava), and both elementals as `F` (a furnace).
    #[test]
    fn glyph_table_is_injective() {
        let mut owner: HashMap<char, String> = HashMap::new();
        owner.insert(PLAYER_GLYPH, "player".to_string());
        for (name, glyph, _) in TILE_GLYPHS {
            if *glyph == ' ' || *glyph == '+' {
                continue; // darkness/out_of_bounds and the grave variants alias on purpose
            }
            if let Some(previous) = owner.insert(*glyph, format!("tile {name}")) {
                panic!("glyph {glyph:?} is used by both {previous} and tile {name}");
            }
        }
        for (kind, glyph) in ENTITY_GLYPHS {
            if *kind == "necromancer" {
                continue; // the boss tile and the boss entity are the same referent
            }
            if let Some(previous) = owner.insert(*glyph, format!("mob {kind}")) {
                panic!("glyph {glyph:?} is used by both {previous} and mob {kind}");
            }
        }
        for glyph in ['!', '*', ':', '-'] {
            if let Some(previous) = owner.insert(glyph, "projectile".to_string()) {
                panic!("glyph {glyph:?} is used by both {previous} and a projectile");
            }
        }
    }

    /// Every mob the generator can spawn must have a glyph, or it renders as
    /// `?` and the map silently loses a threat.
    #[test]
    fn every_spawnable_mob_has_a_glyph() {
        for level in 0..9 {
            for kind in floor_mobs(level) {
                assert_ne!(
                    entity_char(kind),
                    UNKNOWN_GLYPH,
                    "mob {kind} on level {level} has no glyph"
                );
            }
        }
        assert_ne!(entity_char("necromancer"), UNKNOWN_GLYPH);
    }

    /// Every block the generator can place must have a glyph.
    #[test]
    fn every_placed_block_has_a_glyph() {
        for tile in SOLID_BLOCKS {
            assert_ne!(tile_char(tile), UNKNOWN_GLYPH, "block {tile} has no glyph");
        }
        for tile in ["grass", "path", "sand", "gravel", "fire_grass", "ice_grass"] {
            assert_ne!(tile_char(tile), UNKNOWN_GLYPH, "block {tile} has no glyph");
        }
    }

    /// Walkable floors must not share a glyph with each other when `do`
    /// behaves differently on them: `do` on grass can yield a sapling, `do` on
    /// path never can.  Collapsing both to `.` produced 96 dead `do` calls.
    #[test]
    fn grass_and_path_are_distinguishable() {
        assert_ne!(tile_char("grass"), tile_char("path"));
    }

    #[test]
    fn legend_covers_the_glyphs_the_renderer_emits() {
        let legend = glyph_legend();
        for (name, glyph, _) in TILE_GLYPHS {
            if *glyph == ' ' {
                continue;
            }
            assert!(
                legend.contains(&format!("{glyph} ")),
                "legend omits tile {name} ({glyph:?})"
            );
        }
        for (kind, glyph) in ENTITY_GLYPHS {
            if *kind == "necromancer" {
                continue;
            }
            assert!(
                legend.contains(&format!("{glyph} ")),
                "legend omits mob {kind} ({glyph:?})"
            );
        }
    }

    /// A low water density used to compress the whole noise field into the
    /// coast band, erasing the default block and therefore every tree.
    #[test]
    fn low_water_density_keeps_the_default_block() {
        let (water_cut, sand_cut) = water_cuts(0.7, 0.6, 0.05);
        assert!(water_cut > 0.7, "sea cut must rise as water shrinks");
        assert!(sand_cut > 0.6, "coast cut must rise as water shrinks");
        assert!(sand_cut < water_cut, "coast must stay below sea");
        // A mid-field tile stays the default block instead of turning to coast.
        assert!(0.5 < sand_cut);
    }

    #[test]
    fn unit_water_density_is_vanilla() {
        assert_eq!(water_cuts(0.7, 0.6, 1.0), (0.7, 0.6));
    }
}
