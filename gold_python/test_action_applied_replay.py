"""Regression coverage for reconstructing Craftax replay tapes from NEV."""

from __future__ import annotations

import unittest

from gold_python.engine import CraftaxEngine


TASK = {
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
                "learned_spells": ["fireball"],
            },
            "entities": [
                {"kind": "zombie", "pos": [5, 4], "level": 0, "health": 5}
            ],
        },
    },
    "rules": {"base": "symbolic_no_homeostasis"},
}


def _run(actions: list[str]) -> CraftaxEngine:
    engine = CraftaxEngine()
    engine.reset_from_task(TASK)
    for action in actions:
        engine.step(action)
    return engine


class ActionAppliedReplayTests(unittest.TestCase):
    def test_world_advancing_combat_actions_replay_from_action_events(self) -> None:
        submitted_actions = ["do", "shoot_arrow", "cast_spell"]
        original = _run(submitted_actions)
        replay_tape = [
            event["action"]
            for event in original.nev.export()
            if event["kind"] == "action_applied"
        ]

        self.assertEqual(replay_tape, submitted_actions)

        replay = _run(replay_tape)
        self.assertEqual(replay.symbolic_readout(), original.symbolic_readout())


if __name__ == "__main__":
    unittest.main()
