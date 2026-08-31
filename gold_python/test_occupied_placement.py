"""Solid placements must not be allowed beneath active entities."""

from __future__ import annotations

import unittest

from gold_python.engine import CraftaxEngine


TASK = {
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
            ],
        },
    },
    "rules": {"base": "symbolic_no_homeostasis"},
}


class OccupiedPlacementTests(unittest.TestCase):
    def test_table_cannot_be_placed_under_an_entity(self) -> None:
        engine = CraftaxEngine()
        engine.reset_from_task(TASK)
        engine._set_tile((5, 4), "grass")

        engine._place("place_table")

        self.assertEqual(engine.block_at((5, 4)), "grass")
        self.assertEqual(engine.world.inventory["wood"], 2)
        event = engine.nev.export()[-1]
        self.assertEqual(event["transition"], "blocked_by_mob")
        self.assertEqual(event["payload"]["target"], [5, 4])
        self.assertEqual(event["payload"]["entity"]["kind"], "zombie")


if __name__ == "__main__":
    unittest.main()
