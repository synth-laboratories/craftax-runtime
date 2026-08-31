"""Plain-Python Craftax parity vocabulary.

The names in this file mirror the Matthews Craftax full-mode enum surfaces.
They are copied as data, not imported from Craftax/JAX at runtime.
"""

from __future__ import annotations


BLOCK_TYPES = [
    "invalid",
    "out_of_bounds",
    "grass",
    "water",
    "stone",
    "tree",
    "wood",
    "path",
    "coal",
    "iron",
    "diamond",
    "crafting_table",
    "furnace",
    "sand",
    "lava",
    "plant",
    "ripe_plant",
    "wall",
    "darkness",
    "wall_moss",
    "stalagmite",
    "sapphire",
    "ruby",
    "chest",
    "fountain",
    "fire_grass",
    "ice_grass",
    "gravel",
    "fire_tree",
    "ice_shrub",
    "enchantment_table_fire",
    "enchantment_table_ice",
    "necromancer",
    "grave",
    "grave2",
    "grave3",
    "necromancer_vulnerable",
]

ITEM_TYPES = ["none", "torch", "ladder_down", "ladder_up", "ladder_down_blocked"]

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
]

PROJECTILE_TYPES = ["arrow", "dagger", "fireball", "iceball", "arrow2", "slimeball", "fireball2", "iceball2"]
MOB_CLASSES = ["passive", "melee", "ranged", "projectile"]

CRAFTAX_ACHIEVEMENTS = [
    "collect_wood",
    "place_table",
    "eat_cow",
    "collect_sapling",
    "collect_drink",
    "collect_food",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_zombie",
    "collect_stone",
    "place_stone",
    "eat_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_furnace",
    "collect_coal",
    "collect_iron",
    "collect_diamond",
    "make_iron_pickaxe",
    "make_iron_sword",
    "make_arrow",
    "make_torch",
    "place_torch",
    "collect_sapphire",
    "collect_ruby",
    "make_diamond_pickaxe",
    "make_diamond_sword",
    "make_iron_armour",
    "make_diamond_armour",
    "enter_gnomish_mines",
    "enter_dungeon",
    "enter_sewers",
    "enter_vault",
    "enter_troll_mines",
    "enter_fire_realm",
    "enter_ice_realm",
    "enter_graveyard",
    "defeat_gnome_warrior",
    "defeat_gnome_archer",
    "defeat_orc_solider",
    "defeat_orc_mage",
    "defeat_lizard",
    "defeat_kobold",
    "defeat_knight",
    "defeat_archer",
    "defeat_troll",
    "defeat_deep_thing",
    "defeat_pigman",
    "defeat_fire_elemental",
    "defeat_frost_troll",
    "defeat_ice_elemental",
    "damage_necromancer",
    "defeat_necromancer",
    "eat_bat",
    "eat_snail",
    "find_bow",
    "fire_bow",
    "learn_spell",
    "cast_spell",
    "open_chest",
    "drink_potion",
    "enchant_sword",
    "enchant_armour",
]

LEVEL_ACHIEVEMENTS = {
    1: "enter_dungeon",
    2: "enter_gnomish_mines",
    3: "enter_sewers",
    4: "enter_vault",
    5: "enter_troll_mines",
    6: "enter_fire_realm",
    7: "enter_ice_realm",
    8: "enter_graveyard",
}

PASSIVE_MOBS = ["cow", "bat", "snail"]
MELEE_MOBS = ["zombie", "gnome_warrior", "orc_solider", "lizard", "knight", "troll", "pigman", "frost_troll"]
RANGED_MOBS = ["skeleton", "gnome_archer", "orc_mage", "kobold", "archer", "deep_thing", "fire_elemental", "ice_elemental"]
MOB_NAMES = PASSIVE_MOBS + MELEE_MOBS + RANGED_MOBS

MOB_ACHIEVEMENTS = {
    "cow": "eat_cow",
    "bat": "eat_bat",
    "snail": "eat_snail",
    "zombie": "defeat_zombie",
    "skeleton": "defeat_skeleton",
    "gnome_warrior": "defeat_gnome_warrior",
    "gnome_archer": "defeat_gnome_archer",
    "orc_solider": "defeat_orc_solider",
    "orc_mage": "defeat_orc_mage",
    "lizard": "defeat_lizard",
    "kobold": "defeat_kobold",
    "knight": "defeat_knight",
    "archer": "defeat_archer",
    "troll": "defeat_troll",
    "deep_thing": "defeat_deep_thing",
    "pigman": "defeat_pigman",
    "fire_elemental": "defeat_fire_elemental",
    "frost_troll": "defeat_frost_troll",
    "ice_elemental": "defeat_ice_elemental",
}

FLOOR_NAMES = [
    "overworld",
    "dungeon",
    "gnomish_mines",
    "sewers",
    "vault",
    "troll_mines",
    "fire_realm",
    "ice_realm",
    "graveyard",
]

FLOOR_MOBS = {
    0: ("cow", "zombie", "skeleton"),
    1: ("snail", "orc_solider", "orc_mage"),
    2: ("bat", "gnome_warrior", "gnome_archer"),
    3: ("snail", "lizard", "kobold"),
    4: ("snail", "knight", "archer"),
    5: ("bat", "troll", "deep_thing"),
    6: ("bat", "pigman", "fire_elemental"),
    7: ("bat", "frost_troll", "ice_elemental"),
    8: ("cow", "zombie", "skeleton"),
}

FLOOR_MOB_SPAWN_CHANCE = [
    [0.1, 0.02, 0.05, 0.1],
    [0.1, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
    [0.0, 0.06, 0.05, 0.0],
    [0.1, 0.06, 0.05, 0.0],
]

MOB_HEALTH = {
    "cow": 3,
    "bat": 4,
    "snail": 6,
    "zombie": 5,
    "skeleton": 3,
    "gnome_warrior": 7,
    "gnome_archer": 5,
    "orc_solider": 9,
    "orc_mage": 6,
    "lizard": 11,
    "kobold": 8,
    "knight": 12,
    "archer": 12,
    "troll": 20,
    "deep_thing": 4,
    "pigman": 20,
    "fire_elemental": 14,
    "frost_troll": 24,
    "ice_elemental": 16,
    "necromancer": 8,
}

MOB_ALIASES = {
    "goblin": "gnome_warrior",
    "orc_soldier": "orc_solider",
    "knight_archer": "archer",
}

MOB_TYPE_IDS = {
    "cow": 0,
    "bat": 1,
    "snail": 2,
    "zombie": 0,
    "skeleton": 0,
    "gnome_warrior": 1,
    "gnome_archer": 1,
    "orc_solider": 2,
    "orc_mage": 2,
    "lizard": 3,
    "kobold": 3,
    "knight": 4,
    "archer": 4,
    "troll": 5,
    "deep_thing": 5,
    "pigman": 6,
    "fire_elemental": 6,
    "frost_troll": 7,
    "ice_elemental": 7,
}

NO_DAMAGE = [0.0, 0.0, 0.0]
MOB_TYPE_DAMAGE_MAPPING = [
    [NO_DAMAGE, [2.0, 0.0, 0.0], NO_DAMAGE, [2.0, 0.0, 0.0]],
    [NO_DAMAGE, [4.0, 0.0, 0.0], NO_DAMAGE, [4.0, 0.0, 0.0]],
    [NO_DAMAGE, [3.0, 0.0, 0.0], NO_DAMAGE, [0.0, 3.0, 0.0]],
    [NO_DAMAGE, [5.0, 0.0, 0.0], NO_DAMAGE, [0.0, 0.0, 3.0]],
    [NO_DAMAGE, [6.0, 0.0, 0.0], NO_DAMAGE, [5.0, 0.0, 0.0]],
    [NO_DAMAGE, [6.0, 1.0, 1.0], NO_DAMAGE, [4.0, 3.0, 3.0]],
    [NO_DAMAGE, [3.0, 5.0, 0.0], NO_DAMAGE, [3.0, 5.0, 0.0]],
    [NO_DAMAGE, [4.0, 0.0, 5.0], NO_DAMAGE, [4.0, 0.0, 5.0]],
]

NO_DEFENSE = [0.0, 0.0, 0.0]
MOB_TYPE_DEFENSE_MAPPING = [
    [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE],
    [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE],
    [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE],
    [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE],
    [NO_DEFENSE, [0.5, 0.0, 0.0], [0.5, 0.0, 0.0], NO_DEFENSE],
    [NO_DEFENSE, [0.2, 0.0, 0.0], [0.0, 0.0, 0.0], NO_DEFENSE],
    [NO_DEFENSE, [0.9, 1.0, 0.0], [0.9, 1.0, 0.0], NO_DEFENSE],
    [NO_DEFENSE, [0.9, 0.0, 1.0], [0.9, 0.0, 1.0], NO_DEFENSE],
    [NO_DEFENSE, NO_DEFENSE, NO_DEFENSE, NO_DEFENSE],
]

PROJECTILE_DAMAGE_TYPE_IDS = {
    "arrow": 0,
    "dagger": 1,
    "fireball": 2,
    "iceball": 3,
    "arrow2": 4,
    "slimeball": 5,
    "fireball2": 6,
    "iceball2": 7,
}

COLLISION_LAND_CREATURE = [False, True, True]
COLLISION_FLYING = [False, False, False]
COLLISION_AQUATIC = [True, False, True]
COLLISION_AMPHIBIAN = [False, False, True]

MOB_TYPE_COLLISION_MAPPING = [
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING],
    [COLLISION_FLYING, COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_AMPHIBIAN, COLLISION_LAND_CREATURE, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_AQUATIC, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING, COLLISION_FLYING],
    [COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_LAND_CREATURE, COLLISION_FLYING],
]

RANGED_MOB_TYPE_TO_PROJECTILE_TYPE_IDS = [0, 0, 2, 1, 4, 5, 6, 7]

TILE_ALIASES = {
    "table": "crafting_table",
    "torch": "torch",
}
