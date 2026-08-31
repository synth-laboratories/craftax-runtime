# Craftax was not being played on Craftax

**Date:** 2026-08-13
**Scope:** `tasks/craftax-singleplayer` — Rust gold service, Python ReAct container
**Status:** fixed; verified by a live Luna-medium ReAct run on the reference world

## What was wrong

Craftax ships four world presets:

| Preset | Board | Levels | Steps | Densities |
|---|---|---:|---:|---|
| `craftax_default` | 48×48 | 9 | 5000 | `{}` — vanilla |
| `policy_dev_small` | 16×16 | 3 | 120 | `tree: 0.16`, `water: 0.05` |
| `fixture_room` | 9×9 | 2 | 60 | `tree: 0.0`, `water: 0.0` |
| `parity_room` | 11×11 | 9 | 160 | `tree: 0.0`, `water: 0.0` |

Densities are **multipliers on vanilla**, not absolute fractions, so `tree: 0.16`
means 16% of the normal tree count.

`policy_dev_small` or `fixture_room` was the default at **six** independent layers.
Nothing was mislabelled and nothing errored — a caller simply got a toy board and a
number that looked like a Craftax score.

| Layer | Was | Effect |
|---|---|---|
| `craftax_gold.rs` `run_optimizer_rollout` | **hardcoded** `policy_dev_small` | every optimizer / GEPA / GELO / SFT rollout, with no override possible |
| `craftax_gold.rs` `DEFAULT_TASK` | `policy_dev_small` | the interactive `/rollouts` path |
| `craftax_gold.rs` `/dataset` | `policy_dev_small` | the optimizer **task distribution** itself |
| `craftax_gold.rs` `/info` | `policy_dev_small` | what the task advertised about itself |
| `native.rs` scenario resolution | `policy_dev_small` | engine-level default |
| `craftax_singleplayer_container.py` | `policy_dev_template.json` → `fixture_room` | any ReAct caller that omitted `task_path` |

The optimizer path was the worst of these: the world was not a default you could
override, it was a literal in the function body. Every prompt ever optimized against
`/dataset`, and every SFT teacher trajectory collected through `/rollout`, came from a
16×16 board with 16% of the trees and a 120-step budget.

## Why it was invisible

A 16×16 board with 16% tree density and 120 steps cannot reach the early tech tree —
wood → table → wood pickaxe → stone is out of budget. So runs terminated early with
low positive rewards and only tier-1 achievements. That reads exactly like *"the model
is weak at Craftax"*, which is a believable finding, so nobody went looking.

The 2026-08-12 Luna-medium ReAct receipt (`workshop/docs/receipts/2026-08-12/a1.json`)
recorded `mean reward 0.98` over 10 seeds with an achievement union of eight tier-1
entries — `collect_wood`, `collect_sapling`, `collect_food`, `collect_drink`, `eat_cow`,
`place_plant`, `place_table`, `defeat_zombie`. No `collect_stone`, no `make_wood_pickaxe`,
ever. The receipt does not name the world, so the number reads as a Craftax capability
result. Its `actionCount` was exactly 120 — `policy_dev_small`'s step budget.

## The fix

1. **The world is configuration, not a literal.** `run_optimizer_rollout` reads it from
   `/env/config/world`, `/env/world`, `/task/world`, or `policy.config.world`, and
   defaults to `craftax_default`. The dev board is still reachable — by name.
2. **All six defaults are now `craftax_default`.** A toy board requires asking for it.
3. **The step ceiling was raised from 512 to 5000.** The reference world runs to 5000
   steps; the old clamp silently truncated it. Defaults are unchanged, so no existing
   caller shifts by accident.
4. **Every rollout now reports its world.** `summary.world` carries `preset`,
   `max_steps`, `scaled_resources`, the requested spec, and a single
   `is_reference_world` boolean. A receipt can no longer be ambiguous about what was
   played. The Python container reports the same block.
5. **`openrouter` is a first-class provider** and **`reasoning_effort` is a policy knob**,
   recorded in the summary. "Luna medium" is a different policy from "Luna high", so it
   has to be configurable and written down rather than inferred from the provider name.

## Not changed

- `bench.rs` still uses `policy_dev_small` explicitly — it is a throughput benchmark and
  the small board is the point.
- Harbor bundle reference copies under `adapters/harbor/bundles/*/reference/` are
  vendored snapshots.
- Every task template under `tasks/` is untouched; callers that pass `task_path`
  explicitly are unaffected. Only the caller who omitted one changes behaviour, and that
  caller was the one being silently misled.

## Verification

See `WORLD_DEFAULTS_VERIFICATION_2026-08-13.md` for the live Luna-medium ReAct run on
`craftax_default`, compared per-seed against the 2026-08-12 toy-board baseline.
