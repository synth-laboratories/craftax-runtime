#!/usr/bin/env python3
"""Generate train and held-out Craftax policy-sweep suites of distinct worlds.

Mirrors `rogue-singleplayer/scripts/gen_dungeon_suite.py`, but Craftax's suite
schema does not carry the world inline: a Craftax suite is a task template plus
a seed list, and the world is built at `reset()` by `worldgen.generate_world_layout`
from an RNG seeded by `ResolvedTask.seed`. `shared/task_resolve.resolve_task`
does pass the seed through (crafter's does not), so a seed block really is a
block of distinct worlds.

"Really is" is the part that has to be measured rather than asserted. For every
seed this generator resolves the task, resets the engine, and hashes the world
the engine actually built -- terrain maps, item maps, light maps, both ladder
stacks, the spawn point, the potion permutation and the spawned entities. It
then reports how many of those digests are unique, within each split and across
the two, and refuses to write a suite carrying a collision unless
`--allow-duplicates` is passed. A suite whose scenarios are secretly one
scenario resampled is the defect this exists to rule out, and the only honest
way to rule it out is to look.

Held-out seeds are based far from the train block (100_000 by default) so the
two cannot overlap even if either is widened later.

    python3 scripts/gen_world_suite.py --n 100 --heldout 100 --max-steps 500 \
        --out defaults/policy_sweep/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

TASK_DIR = Path(__file__).resolve().parents[1]
for _path in (TASK_DIR, TASK_DIR / "gold_python", TASK_DIR / "shared"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from task_resolve import resolve_task  # noqa: E402
from gold_python.engine import CraftaxEngine  # noqa: E402

SUITE_SCHEMA = "gamebench.craftax.policy_sweep.v1"
WORLD_DIGEST_ALGORITHM = "sha256/craftax_world_layout.v1"

# Train seeds start at 101 to stay compatible with the hand-authored suites that
# already used that block; held-out starts far enough away that widening either
# split by four orders of magnitude still cannot make them meet.
TRAIN_SEED_BASE = 100
HELDOUT_SEED_BASE = 100_000


def world_digest(task_template: str, seed: int) -> str:
    """Hash the world the engine actually builds for `seed`.

    Hashing the suite entry would prove nothing -- the seeds are distinct by
    construction. What has to be distinct is the world downstream of the seed.
    """

    task = json.loads((TASK_DIR / task_template).read_text())
    task["seed"] = seed
    task["task_id"] = f"{task.get('task_id', 'craftax')}_{seed}"
    resolved = resolve_task(task, seed)
    engine = CraftaxEngine()
    engine.reset(resolved)
    world = engine.world
    payload = {
        "maps": world.maps,
        "item_maps": world.item_maps,
        "light_maps": world.light_maps,
        "down_ladders": [list(pos) for pos in world.down_ladders],
        "up_ladders": [list(pos) for pos in world.up_ladders],
        "player_pos": list(world.player_pos),
        "potion_mapping": list(world.potion_mapping),
        "entities": [entity.to_dict() for entity in world.entities],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_suite(
    seeds: list[int],
    *,
    suite_id: str,
    split: str,
    task_template: str,
    max_steps: int,
    digests: dict[int, str],
) -> dict[str, Any]:
    return {
        "schema": SUITE_SCHEMA,
        "suite_id": suite_id,
        "split": split,
        "task_template": task_template,
        "seeds": seeds,
        "max_steps": max_steps,
        "generator": {
            "kind": "gold_python.worldgen.generate_world_layout",
            "script": "scripts/gen_world_suite.py",
            "task_template": task_template,
            "max_steps": max_steps,
            "seed_base": seeds[0] - 1 if seeds else 0,
            "world_digest_algorithm": WORLD_DIGEST_ALGORITHM,
        },
        # Provenance, not input: the sweep reads only `seeds`, `task_template`
        # and `max_steps`. These let anyone re-derive the uniqueness claim
        # without re-running the generator.
        "unique_world_count": len({digests[seed] for seed in seeds}),
        "world_digests": {str(seed): digests[seed][:16] for seed in seeds},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="training worlds")
    parser.add_argument("--heldout", type=int, default=100, help="held-out worlds")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="episode step budget; must be large enough that the reference "
             "baseline can act, or true reward degenerates to a constant",
    )
    parser.add_argument("--task-template", default="tasks/craftax_default_template.json")
    parser.add_argument("--train-seed-base", type=int, default=TRAIN_SEED_BASE)
    parser.add_argument("--heldout-seed-base", type=int, default=HELDOUT_SEED_BASE)
    parser.add_argument("--out", required=True, help="directory to write suites into")
    parser.add_argument("--prefix", default="craftax_gen")
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="write the suites even if two seeds produce the same world",
    )
    args = parser.parse_args()

    if args.heldout_seed_base <= args.train_seed_base + args.n:
        raise SystemExit(
            "held-out seed base overlaps the train block: "
            f"train ends at {args.train_seed_base + args.n}, "
            f"held-out starts at {args.heldout_seed_base + 1}"
        )

    train_seeds = list(range(args.train_seed_base + 1, args.train_seed_base + args.n + 1))
    heldout_seeds = list(
        range(args.heldout_seed_base + 1, args.heldout_seed_base + args.heldout + 1)
    )
    overlap = set(train_seeds) & set(heldout_seeds)
    if overlap:
        raise SystemExit(f"train and held-out seed blocks overlap: {sorted(overlap)[:5]}")

    digests: dict[int, str] = {}
    for seed in train_seeds + heldout_seeds:
        digests[seed] = world_digest(args.task_template, seed)

    train_unique = len({digests[s] for s in train_seeds})
    heldout_unique = len({digests[s] for s in heldout_seeds})
    all_seeds = train_seeds + heldout_seeds
    combined_unique = len({digests[s] for s in all_seeds})
    cross = {digests[s] for s in train_seeds} & {digests[s] for s in heldout_seeds}

    print(f"world digest algorithm: {WORLD_DIGEST_ALGORITHM}")
    print(f"train   : {train_unique}/{len(train_seeds)} unique worlds")
    print(f"heldout : {heldout_unique}/{len(heldout_seeds)} unique worlds")
    print(f"combined: {combined_unique}/{len(all_seeds)} unique worlds")
    print(f"train/heldout shared worlds: {len(cross)}")

    degenerate = combined_unique != len(all_seeds)
    if degenerate and not args.allow_duplicates:
        raise SystemExit(
            "refusing to write: generated worlds are not all distinct "
            f"({combined_unique}/{len(all_seeds)} unique). A suite of N seeds "
            "over one world measures one scenario N times."
        )

    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    written: list[tuple[Path, int, int]] = []
    for name, split, seeds in (
        (f"{args.prefix}_train_v1", "train", train_seeds),
        (f"{args.prefix}_heldout_v1", "heldout", heldout_seeds),
    ):
        suite = build_suite(
            seeds,
            suite_id=name,
            split=split,
            task_template=args.task_template,
            max_steps=args.max_steps,
            digests=digests,
        )
        path = out / f"{name}.json"
        path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append((path, len(seeds), int(suite["unique_world_count"])))
    for path, count, unique in written:
        print(f"{path}  ({count} worlds, {unique} unique)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
