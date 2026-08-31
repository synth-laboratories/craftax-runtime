"""Deterministic no-JAX Craftax world generation.

This mirrors the upstream Craftax level stack and configuration data in plain
Python. It intentionally has no JAX or upstream Craftax runtime dependency.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


NUM_ROOMS = 8
MIN_ROOM_SIZE = 5
MAX_ROOM_SIZE = 10
CHUNK_SIZE = 16


@dataclass(frozen=True)
class SmoothGenConfig:
    default_block: str
    sea_block: str
    coast_block: str
    mountain_block: str
    path_block: str
    inner_mountain_block: str
    ore_requirement_blocks: tuple[str, ...]
    ores: tuple[str, ...]
    ore_chances: tuple[float, ...]
    tree_requirement_block: str
    tree: str
    lava: str
    player_spawn: str
    valid_ladder: str
    ladder_up: bool
    ladder_down: bool
    player_proximity_map_water_strength: int
    player_proximity_map_water_max: float
    player_proximity_map_mountain_strength: int
    player_proximity_map_mountain_max: float
    default_light: float
    water_threshold: float
    sand_threshold: float
    tree_threshold_uniform: float
    tree_threshold_perlin: float


@dataclass(frozen=True)
class DungeonConfig:
    special_block: str
    fountain_block: str
    rare_path_replacement_block: str
    valid_ladder: str


@dataclass(frozen=True)
class WorldLayout:
    maps: list[list[list[str]]]
    item_maps: list[list[list[str]]]
    light_maps: list[list[list[float]]]
    down_ladders: list[tuple[int, int]]
    up_ladders: list[tuple[int, int]]
    player_pos: tuple[int, int]


OVERWORLD_CONFIG = SmoothGenConfig(
    default_block="grass",
    sea_block="water",
    coast_block="sand",
    mountain_block="stone",
    path_block="path",
    inner_mountain_block="path",
    ore_requirement_blocks=("stone", "stone", "stone", "stone", "stone"),
    ores=("coal", "iron", "diamond", "out_of_bounds", "out_of_bounds"),
    ore_chances=(0.03, 0.02, 0.001, 0.0, 0.0),
    tree_requirement_block="grass",
    tree="tree",
    lava="lava",
    player_spawn="grass",
    valid_ladder="path",
    ladder_up=False,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=5,
    player_proximity_map_mountain_max=1.0,
    default_light=1.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

GNOMISH_MINES_CONFIG = SmoothGenConfig(
    default_block="path",
    sea_block="water",
    coast_block="path",
    mountain_block="stone",
    path_block="stone",
    inner_mountain_block="stone",
    ore_requirement_blocks=("stone", "stone", "stone", "stone", "stone"),
    ores=("coal", "iron", "diamond", "sapphire", "ruby"),
    ore_chances=(0.04, 0.02, 0.005, 0.0025, 0.0025),
    tree_requirement_block="path",
    tree="stalagmite",
    lava="lava",
    player_spawn="path",
    valid_ladder="path",
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

TROLL_MINES_CONFIG = SmoothGenConfig(
    default_block="path",
    sea_block="water",
    coast_block="path",
    mountain_block="stone",
    path_block="stone",
    inner_mountain_block="stone",
    ore_requirement_blocks=("stone", "stone", "stone", "stone", "stone"),
    ores=("coal", "iron", "diamond", "sapphire", "ruby"),
    ore_chances=(0.04, 0.03, 0.01, 0.01, 0.01),
    tree_requirement_block="path",
    tree="stalagmite",
    lava="lava",
    player_spawn="path",
    valid_ladder="path",
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

FIRE_LEVEL_CONFIG = SmoothGenConfig(
    default_block="fire_grass",
    sea_block="lava",
    coast_block="sand",
    mountain_block="stone",
    path_block="stone",
    inner_mountain_block="stone",
    ore_requirement_blocks=("stone", "stone", "stone", "stone", "stone"),
    ores=("coal", "iron", "diamond", "sapphire", "ruby"),
    ore_chances=(0.05, 0.0, 0.0, 0.0, 0.025),
    tree_requirement_block="fire_grass",
    tree="fire_tree",
    lava="lava",
    player_spawn="fire_grass",
    valid_ladder="fire_grass",
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=5,
    player_proximity_map_mountain_max=1.0,
    default_light=1.0,
    water_threshold=0.5,
    sand_threshold=0.6,
    tree_threshold_uniform=0.8,
    tree_threshold_perlin=0.5,
)

ICE_LEVEL_CONFIG = SmoothGenConfig(
    default_block="ice_grass",
    sea_block="water",
    coast_block="ice_grass",
    mountain_block="stone",
    path_block="stone",
    inner_mountain_block="stone",
    ore_requirement_blocks=("stone", "stone", "stone", "stone", "stone"),
    ores=("coal", "iron", "diamond", "sapphire", "ruby"),
    ore_chances=(0.0, 0.0, 0.005, 0.02, 0.0),
    tree_requirement_block="ice_grass",
    tree="ice_shrub",
    lava="water",
    player_spawn="ice_grass",
    valid_ladder="ice_grass",
    ladder_up=True,
    ladder_down=True,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=17,
    player_proximity_map_mountain_max=1.5,
    default_light=0.0,
    water_threshold=0.5,
    sand_threshold=0.6,
    tree_threshold_uniform=0.4,
    tree_threshold_perlin=0.5,
)

BOSS_LEVEL_CONFIG = SmoothGenConfig(
    default_block="path",
    sea_block="path",
    coast_block="path",
    mountain_block="wall",
    path_block="wall",
    inner_mountain_block="wall",
    ore_requirement_blocks=("wall", "grave", "grave", "wall", "wall"),
    ores=("wall_moss", "grave2", "grave3", "sapphire", "ruby"),
    ore_chances=(0.1, 0.333, 0.5, 0.0, 0.0),
    tree_requirement_block="path",
    tree="grave",
    lava="wall",
    player_spawn="necromancer",
    valid_ladder="path",
    ladder_up=False,
    ladder_down=False,
    player_proximity_map_water_strength=5,
    player_proximity_map_water_max=1.0,
    player_proximity_map_mountain_strength=10,
    player_proximity_map_mountain_max=10.0,
    default_light=0.0,
    water_threshold=0.7,
    sand_threshold=0.6,
    tree_threshold_uniform=0.95,
    tree_threshold_perlin=-1.0,
)

DUNGEON_CONFIG = DungeonConfig(
    special_block="path",
    fountain_block="fountain",
    rare_path_replacement_block="path",
    valid_ladder="path",
)
SEWER_CONFIG = DungeonConfig(
    special_block="enchantment_table_ice",
    fountain_block="water",
    rare_path_replacement_block="water",
    valid_ladder="path",
)
VAULTS_CONFIG = DungeonConfig(
    special_block="enchantment_table_fire",
    fountain_block="fountain",
    rare_path_replacement_block="path",
    valid_ladder="path",
)

LEVEL_STACK: tuple[tuple[str, SmoothGenConfig | DungeonConfig], ...] = (
    ("smooth", OVERWORLD_CONFIG),
    ("dungeon", DUNGEON_CONFIG),
    ("smooth", GNOMISH_MINES_CONFIG),
    ("dungeon", SEWER_CONFIG),
    ("dungeon", VAULTS_CONFIG),
    ("smooth", TROLL_MINES_CONFIG),
    ("smooth", FIRE_LEVEL_CONFIG),
    ("smooth", ICE_LEVEL_CONFIG),
    ("smooth", BOSS_LEVEL_CONFIG),
)


def generate_world_layout(
    width: int,
    height: int,
    levels: int,
    rng: random.Random,
    densities: dict[str, Any] | None = None,
) -> WorldLayout:
    densities = densities or {}
    player_pos = (width // 2, height // 2)
    maps: list[list[list[str]]] = []
    item_maps: list[list[list[str]]] = []
    light_maps: list[list[list[float]]] = []
    down_ladders: list[tuple[int, int]] = []
    up_ladders: list[tuple[int, int]] = []

    for level in range(levels):
        kind, config = LEVEL_STACK[min(level, len(LEVEL_STACK) - 1)]
        if kind == "smooth":
            assert isinstance(config, SmoothGenConfig)
            generated = _generate_smooth_level(width, height, rng, player_pos, config, densities)
        else:
            assert isinstance(config, DungeonConfig)
            generated = _generate_dungeon_level(width, height, rng, config)
        maps.append(generated[0])
        item_maps.append(generated[1])
        light_maps.append(generated[2])
        down_ladders.append(generated[3])
        up_ladders.append(generated[4])

    return WorldLayout(
        maps=maps,
        item_maps=item_maps,
        light_maps=light_maps,
        down_ladders=down_ladders,
        up_ladders=up_ladders,
        player_pos=player_pos,
    )


def _generate_smooth_level(
    width: int,
    height: int,
    rng: random.Random,
    player_pos: tuple[int, int],
    config: SmoothGenConfig,
    densities: dict[str, Any],
) -> tuple[list[list[str]], list[list[str]], list[list[float]], tuple[int, int], tuple[int, int]]:
    larger_res = (max(1, width // 4), max(1, height // 4))
    small_res = (max(1, width // 16), max(1, height // 16))
    x_res = (max(1, width // 8), max(1, height // 2))
    water_noise = _fractal_noise_2d(rng, width, height, small_res)
    mountain_noise = _fractal_noise_2d(rng, width, height, small_res)
    path_x = _fractal_noise_2d(rng, width, height, x_res)
    tree_noise = _fractal_noise_2d(rng, width, height, larger_res)

    grid = [[config.default_block for _ in range(width)] for _ in range(height)]
    water_density = _density(densities, "water", 1.0)
    tree_density = _density(densities, "tree", 1.0)
    mountain_threshold = 0.7
    for y in range(height):
        for x in range(width):
            distance = math.sqrt((x - player_pos[0]) ** 2 + (y - player_pos[1]) ** 2)
            water_proximity = min(
                config.player_proximity_map_water_max,
                distance / max(1, config.player_proximity_map_water_strength),
            )
            water = water_noise[y][x] + water_proximity - 1.0
            if water_density <= 0.0:
                water = -1.0
            water_cut, sand_cut = _water_cuts(
                config.water_threshold, config.sand_threshold, water_density
            )

            block = config.sea_block if water > water_cut else config.default_block
            if water > sand_cut and block != config.sea_block:
                block = config.coast_block

            mountain_proximity = min(
                config.player_proximity_map_mountain_max,
                distance / max(1, config.player_proximity_map_mountain_strength),
            )
            mountain = mountain_noise[y][x] + 0.05 + mountain_proximity - 1.0
            if mountain > mountain_threshold:
                block = config.mountain_block

            path_y = path_x[x][y] if x < height and y < width else path_x[y][x]
            if mountain > mountain_threshold and (path_x[y][x] > 0.8 or path_y > 0.8):
                block = config.path_block
            if mountain > 0.85 and water > 0.4:
                block = config.inner_mountain_block

            if (
                tree_density > 0.0
                and block == config.tree_requirement_block
                and tree_noise[y][x] > config.tree_threshold_perlin
                and rng.random() > _tree_uniform_threshold(config, tree_density)
            ):
                block = config.tree

            grid[y][x] = block

    for y in range(height):
        for x in range(width):
            if mountain_noise[y][x] + 0.05 > 0.85 and tree_noise[y][x] > 0.7:
                if grid[y][x] in {config.mountain_block, config.inner_mountain_block, config.tree}:
                    grid[y][x] = config.lava

    for req, ore, chance in zip(config.ore_requirement_blocks, config.ores, config.ore_chances, strict=True):
        if chance <= 0.0 or ore == "out_of_bounds":
            continue
        for y in range(height):
            for x in range(width):
                if grid[y][x] == req and rng.random() < chance:
                    grid[y][x] = ore

    px, py = player_pos
    if 0 <= px < width and 0 <= py < height:
        grid[py][px] = config.player_spawn

    item_map = _empty_item_map(width, height)
    light_map = [[config.default_light for _ in range(width)] for _ in range(height)]
    down_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, player_pos)
    up_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, down_ladder)
    if config.ladder_down:
        item_map[down_ladder[1]][down_ladder[0]] = "ladder_down"
    if config.ladder_up:
        item_map[up_ladder[1]][up_ladder[0]] = "ladder_up"
        _add_ladder_light(light_map, up_ladder)
    return grid, item_map, light_map, down_ladder, up_ladder


def _generate_dungeon_level(
    width: int,
    height: int,
    rng: random.Random,
    config: DungeonConfig,
) -> tuple[list[list[str]], list[list[str]], list[list[float]], tuple[int, int], tuple[int, int]]:
    if width < MIN_ROOM_SIZE * 2 or height < MIN_ROOM_SIZE * 2:
        return _generate_small_dungeon(width, height, rng, config)

    grid = [["wall" for _ in range(width)] for _ in range(height)]
    item_map = _empty_item_map(width, height)
    rooms: list[tuple[int, int, int, int]] = []
    chunks_x = max(1, width // CHUNK_SIZE)
    chunks_y = max(1, height // CHUNK_SIZE)
    chunks = [(cx, cy) for cy in range(chunks_y) for cx in range(chunks_x)]
    rng.shuffle(chunks)

    for room_index in range(min(NUM_ROOMS, len(chunks))):
        cx, cy = chunks[room_index]
        room_w = rng.randrange(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        room_h = rng.randrange(MIN_ROOM_SIZE, MAX_ROOM_SIZE)
        origin_x = cx * CHUNK_SIZE + rng.randrange(0, max(1, CHUNK_SIZE - MIN_ROOM_SIZE))
        origin_y = cy * CHUNK_SIZE + rng.randrange(0, max(1, CHUNK_SIZE - MIN_ROOM_SIZE))
        x0 = min(max(1, origin_x), max(1, width - room_w - 1))
        y0 = min(max(1, origin_y), max(1, height - room_h - 1))
        room_w = min(room_w, width - x0 - 1)
        room_h = min(room_h, height - y0 - 1)
        rooms.append((x0, y0, room_w, room_h))
        for y in range(y0, y0 + room_h):
            for x in range(x0, x0 + room_w):
                grid[y][x] = "path"
        for tx, ty in ((x0, y0), (x0 + room_w - 1, y0), (x0, y0 + room_h - 1), (x0 + room_w - 1, y0 + room_h - 1)):
            item_map[ty][tx] = "torch"
        if room_w > 2 and room_h > 2:
            chest = (rng.randrange(x0 + 1, x0 + room_w - 1), rng.randrange(y0 + 1, y0 + room_h - 1))
            grid[chest[1]][chest[0]] = "chest"
            item_map[chest[1]][chest[0]] = "none"
            if rng.random() > 0.5:
                fountain = (rng.randrange(x0 + 1, x0 + room_w - 1), rng.randrange(y0 + 1, y0 + room_h - 1))
                grid[fountain[1]][fountain[0]] = config.fountain_block
                item_map[fountain[1]][fountain[0]] = "none"

    if not rooms:
        return _generate_small_dungeon(width, height, rng, config)

    included = [rooms[-1]]
    for room in rooms:
        sink = rng.choice(included)
        _carve_corridor(grid, _room_anchor(room), _room_anchor(sink))
        if room not in included:
            included.append(room)

    special_room = rooms[0]
    sx = min(width - 2, special_room[0] + 2)
    sy = min(height - 2, special_room[1] + 2)
    grid[sy][sx] = config.special_block
    item_map[sy][sx] = "none"

    grid = _apply_dungeon_visuals(grid, item_map, rng, config)
    down_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, (-1, -1))
    item_map[down_ladder[1]][down_ladder[0]] = "ladder_down"
    up_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, down_ladder)
    item_map[up_ladder[1]][up_ladder[0]] = "ladder_up"
    light_map = [[1.0 for _ in range(width)] for _ in range(height)]
    return grid, item_map, light_map, down_ladder, up_ladder


def _generate_small_dungeon(
    width: int,
    height: int,
    rng: random.Random,
    config: DungeonConfig,
) -> tuple[list[list[str]], list[list[str]], list[list[float]], tuple[int, int], tuple[int, int]]:
    grid = [["wall" for _ in range(width)] for _ in range(height)]
    item_map = _empty_item_map(width, height)
    for y in range(1, max(1, height - 1)):
        for x in range(1, max(1, width - 1)):
            grid[y][x] = "path"
    if width > 4 and height > 4:
        grid[2][2] = config.special_block
    down_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, (-1, -1))
    item_map[down_ladder[1]][down_ladder[0]] = "ladder_down"
    up_ladder = _choose_ladder(grid, item_map, config.valid_ladder, rng, down_ladder)
    item_map[up_ladder[1]][up_ladder[0]] = "ladder_up"
    light_map = [[1.0 for _ in range(width)] for _ in range(height)]
    return grid, item_map, light_map, down_ladder, up_ladder


def _fractal_noise_2d(
    rng: random.Random,
    width: int,
    height: int,
    res: tuple[int, int],
) -> list[list[float]]:
    values = _perlin_noise_2d(rng, width, height, res)
    flat = [value for row in values for value in row]
    lo = min(flat)
    hi = max(flat)
    if math.isclose(lo, hi):
        return [[0.5 for _ in range(width)] for _ in range(height)]
    scale = hi - lo
    return [[(value - lo) / scale for value in row] for row in values]


def _perlin_noise_2d(
    rng: random.Random,
    width: int,
    height: int,
    res: tuple[int, int],
) -> list[list[float]]:
    res_x = max(1, res[0])
    res_y = max(1, res[1])
    angles = [[2.0 * math.pi * rng.random() for _ in range(res_x + 1)] for _ in range(res_y + 1)]
    gradients = [[(math.cos(angle), math.sin(angle)) for angle in row] for row in angles]
    values: list[list[float]] = []
    for y in range(height):
        row: list[float] = []
        gy = (y / max(1, height)) * res_y
        iy = min(res_y - 1, int(math.floor(gy)))
        fy = gy - iy
        for x in range(width):
            gx = (x / max(1, width)) * res_x
            ix = min(res_x - 1, int(math.floor(gx)))
            fx = gx - ix
            g00 = gradients[iy][ix]
            g10 = gradients[iy][ix + 1]
            g01 = gradients[iy + 1][ix]
            g11 = gradients[iy + 1][ix + 1]
            n00 = _dot(g00, fx, fy)
            n10 = _dot(g10, fx - 1.0, fy)
            n01 = _dot(g01, fx, fy - 1.0)
            n11 = _dot(g11, fx - 1.0, fy - 1.0)
            sx = _fade(fx)
            sy = _fade(fy)
            n0 = _lerp(n00, n10, sx)
            n1 = _lerp(n01, n11, sx)
            row.append(math.sqrt(2.0) * _lerp(n0, n1, sy))
        values.append(row)
    return values


def _apply_dungeon_visuals(
    grid: list[list[str]],
    item_map: list[list[str]],
    rng: random.Random,
    config: DungeonConfig,
) -> list[list[str]]:
    height = len(grid)
    width = len(grid[0]) if height else 0
    out = [[cell for cell in row] for row in grid]
    for y in range(height):
        for x in range(width):
            adjacent_path = _has_cross_neighbor(grid, x, y, lambda block: block != "wall")
            rare = rng.random() < 0.1
            if grid[y][x] == "wall":
                out[y][x] = "wall_moss" if adjacent_path and rare else ("wall" if adjacent_path else "darkness")
            elif rare and grid[y][x] == "path" and item_map[y][x] == "none":
                out[y][x] = config.rare_path_replacement_block
    return out


def _choose_ladder(
    grid: list[list[str]],
    item_map: list[list[str]],
    valid_block: str,
    rng: random.Random,
    avoid: tuple[int, int],
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for y, row in enumerate(grid):
        for x, block in enumerate(row):
            if (x, y) == avoid:
                continue
            if block == valid_block and item_map[y][x] == "none":
                candidates.append((x, y))
    if not candidates:
        for y, row in enumerate(grid):
            for x, block in enumerate(row):
                if (x, y) != avoid and block not in {"wall", "darkness", "water", "lava"}:
                    candidates.append((x, y))
    if not candidates:
        return (max(0, min(len(grid[0]) - 1, 1)), max(0, min(len(grid) - 1, 1)))
    return rng.choice(candidates)


def _add_ladder_light(light_map: list[list[float]], pos: tuple[int, int]) -> None:
    cx, cy = pos
    height = len(light_map)
    width = len(light_map[0]) if height else 0
    for y in range(max(0, cy - 4), min(height, cy + 5)):
        for x in range(max(0, cx - 4), min(width, cx + 5)):
            distance = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            light_map[y][x] = max(light_map[y][x], max(0.0, min(1.0, 1.0 - distance / 5.0)))


def _carve_corridor(grid: list[list[str]], source: tuple[int, int], sink: tuple[int, int]) -> None:
    sx, sy = source
    tx, ty = sink
    step_x = 1 if tx >= sx else -1
    for x in range(sx, tx + step_x, step_x):
        if grid[sy][x] == "wall":
            grid[sy][x] = "path"
    step_y = 1 if ty >= sy else -1
    for y in range(sy, ty + step_y, step_y):
        if grid[y][tx] == "wall":
            grid[y][tx] = "path"


def _room_anchor(room: tuple[int, int, int, int]) -> tuple[int, int]:
    return room[0], room[1]


def _empty_item_map(width: int, height: int) -> list[list[str]]:
    return [["none" for _ in range(width)] for _ in range(height)]


def _has_cross_neighbor(grid: list[list[str]], x: int, y: int, predicate: Any) -> bool:
    height = len(grid)
    width = len(grid[0]) if height else 0
    for nx, ny in ((x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
        if 0 <= nx < width and 0 <= ny < height and predicate(grid[ny][nx]):
            return True
    return False


def _tree_uniform_threshold(config: SmoothGenConfig, density: float) -> float:
    chance = max(0.0, 1.0 - config.tree_threshold_uniform) * max(0.0, density)
    return max(0.0, min(1.0, 1.0 - chance))


def _density(densities: dict[str, Any], key: str, default: float) -> float:
    """Scale a feature relative to the vanilla amount.

    ``1.0`` is vanilla, ``0.0`` removes the feature, ``0.5`` is half as much.
    It is never an absolute fraction of the map.
    """

    try:
        return max(0.0, float(densities.get(key, default)))
    except (TypeError, ValueError):
        return default


def _water_cuts(
    water_threshold: float, sand_threshold: float, density: float
) -> tuple[float, float]:
    """Scale the sea and coast cut points instead of distorting the field.

    Compressing the noise toward ``water_threshold`` (the previous behaviour)
    pulled every tile into the narrow band between ``sand_threshold`` and
    ``water_threshold``, so a low density turned the whole surface into coast
    and erased the default block.  Trees require the default block, so
    ``water: 0.05`` silently produced a treeless world on every seed.
    """

    if density == 1.0:
        return water_threshold, sand_threshold

    def shrink(cut: float) -> float:
        if density >= 1.0:
            return max(-1.0, min(1.0, cut - (density - 1.0) * (cut + 1.0)))
        return max(-1.0, min(1.0, cut + (1.0 - density) * (1.0 - cut)))

    return shrink(water_threshold), shrink(sand_threshold)


def _dot(gradient: tuple[float, float], x: float, y: float) -> float:
    return gradient[0] * x + gradient[1] * y


def _fade(value: float) -> float:
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, weight: float) -> float:
    return a * (1.0 - weight) + b * weight
