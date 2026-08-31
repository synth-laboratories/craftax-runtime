"""The two engines must render the same world the same way.

`gold_python` and `gold_rust` each carry their own glyph table.  They had
already drifted (`fire_tree` was `Y` in Python and `T` in Rust), and the Rust
side derived mob glyphs from `kind[0].to_ascii_uppercase()`, which collided
with the player, trees, lava, and the furnace.  These tests parse the Rust
table directly so a change on either side fails here.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent.parent
RUST_SOURCE = TASK_DIR / "gold_rust" / "src" / "native.rs"
PYTHON_SOURCE = TASK_DIR / "gold_python" / "engine.py"

_RUST_TILE_ROW = re.compile(r'\(\s*"([a-z_0-9]+)"\s*,\s*\'(\\?.)\'\s*,')
_RUST_ENTITY_ROW = re.compile(r'\(\s*"([a-z_0-9]+)"\s*,\s*\'(\\?.)\'\s*\)')


def _unescape(glyph: str) -> str:
    return {"\\'": "'", "\\\\": "\\"}.get(glyph, glyph)


def _rust_block(name: str) -> str:
    source = RUST_SOURCE.read_text()
    start = source.index(f"pub const {name}")
    return source[start : source.index("];", start)]


def _rust_tiles() -> dict[str, str]:
    return {
        tile: _unescape(glyph)
        for tile, glyph in _RUST_TILE_ROW.findall(_rust_block("TILE_GLYPHS"))
    }


def _rust_entities() -> dict[str, str]:
    return {
        kind: _unescape(glyph)
        for kind, glyph in _RUST_ENTITY_ROW.findall(_rust_block("ENTITY_GLYPHS"))
    }


def _python_table(name: str) -> dict[str, str]:
    tree = ast.parse(PYTHON_SOURCE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {PYTHON_SOURCE}")


def test_tile_glyphs_match_across_engines() -> None:
    assert _python_table("TILE_GLYPHS") == _rust_tiles()


def test_entity_glyphs_match_across_engines() -> None:
    assert _python_table("ENTITY_GLYPHS") == _rust_entities()


def test_glyphs_are_injective() -> None:
    tiles = _python_table("TILE_GLYPHS")
    entities = _python_table("ENTITY_GLYPHS")

    owner: dict[str, str] = {"P": "player"}
    for tile, glyph in tiles.items():
        # darkness/out_of_bounds share blank, and the grave variants are one
        # thing rendered three ways.
        if glyph in {" ", "+"}:
            continue
        assert glyph not in owner, f"{glyph!r}: {owner[glyph]} vs tile {tile}"
        owner[glyph] = f"tile {tile}"
    for kind, glyph in entities.items():
        if kind == "necromancer":
            continue  # the boss tile and the boss entity are the same referent
        assert glyph not in owner, f"{glyph!r}: {owner[glyph]} vs mob {kind}"
        owner[glyph] = f"mob {kind}"
    for glyph in ("!", "*", ":", "-"):
        assert glyph not in owner, f"{glyph!r}: {owner[glyph]} vs a projectile"


def test_walkable_floors_with_different_do_behaviour_are_distinct() -> None:
    tiles = _python_table("TILE_GLYPHS")
    # `do` on grass can yield a sapling; `do` on path never can.  Collapsing
    # both to "." produced dead `do` calls the policy could not diagnose.
    assert tiles["grass"] != tiles["path"]
