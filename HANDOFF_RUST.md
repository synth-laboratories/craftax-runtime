# Handoff - Craftax Rust Gold Lane

Status: implemented through Rust fixture parity, HTTP, checkpoint restore, perf bench, and policy smoke support.
Date: 2026-06-20
Task root: `tasks/craftax-singleplayer/`

## Current State

Craftax now has two local GameBench gold lanes:

| Lane | Role | Local port |
|------|------|------------|
| Python `gold_python/` | Authority/debug/policy iteration | 8097 |
| Rust `gold_rust/` | Native throughput lane and HTTP service | 8098 |

The Rust lane is a native port, not a wrapper. It does not import JAX, Crafter, or the Python engine at runtime.

Implemented Rust surfaces:

- Resolver parity: `config_hash`, `episode_id`, and resolved task output.
- Full scenario runner: `gold_rust/src/bin/scenario.rs`.
- Native engine/action path for all 42 Craftax actions.
- Structured and legacy NEV output.
- Grid hash parity for all gold fixtures.
- JSON checkpoint encode/restore.
- Restore equivalence check binary.
- HTTP service: `gold_rust/src/bin/craftax_gold.rs`.
- Perf bench: `gold_rust/src/bin/bench.rs`.
- Policy-ready readouts: `observation`, `observation_text`, `valid_actions`, ASCII state.

## Files Added Or Changed

- `gold_rust/`
  - `Cargo.toml`
  - `src/lib.rs`
  - `src/native.rs`
  - `src/bin/scenario.rs`
  - `src/bin/craftax_gold.rs`
  - `src/bin/bench.rs`
  - `src/bin/restore_check.rs`
- `scripts/compare_gold_lanes.py`
- `scripts/run_policy_sweep.py`
  - Added `--lane rust --base-url http://127.0.0.1:8098`.
- `scripts/run_service.py`
- `scripts/verify_import_boundary.py`
- `shared/http_contract.md`
- `task.yaml`
  - `gold_lane: python+rust`
- `README.md`
- `policies/heuristic_baseline.py`
  - Improved deterministic policy path for wood -> crafting table -> wood pickaxe.
  - Uses lightweight ASCII-grid BFS for table/resource routing.
- `tasks/policy_dev_template.json`
  - Expanded policy dev room so the smoke fixture can validate real pickaxe crafting costs.

Note: the whole `tasks/craftax-singleplayer/` tree is currently untracked in this checkout.

## Verified Commands

Rust build:

```bash
cargo build --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml
```

Restore equivalence:

```bash
cargo run --quiet --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml --bin restore_check
```

Last known result:

```text
RESTORE_EQUIVALENCE_OK scenarios=36 checks=200
```

Resolver parity:

```bash
python3 tasks/craftax-singleplayer/scripts/compare_gold_lanes.py --resolver-only
```

Fixture parity:

```bash
python3 tasks/craftax-singleplayer/scripts/compare_gold_lanes.py --parity-only
```

Last known result:

```text
match=true, scenarios=36/36
```

Perf:

```bash
python3 tasks/craftax-singleplayer/scripts/compare_gold_lanes.py --parity-only --perf
```

Last observed perf from prior pass:

- checkpoint p50: about 103 KB
- restore p50: about 0.42 ms
- Rust native throughput: about 93k steps/sec in the direct bench path

Import boundary:

```bash
PYTHONPATH=tasks/craftax-singleplayer/gold_python:tasks/craftax-singleplayer/shared \
python3 tasks/craftax-singleplayer/scripts/verify_import_boundary.py
```

Last known result: OK.

## HTTP Service

Start Rust service:

```bash
cargo run --release --quiet \
  --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml \
  --bin craftax_gold -- --host 127.0.0.1 --port 8098
```

Useful routes:

- `GET /health`
- `GET /info`
- `POST /run_scenario`
- `POST /rollouts`
- `GET /rollouts/{rollout_id}/readout`
- `POST /rollouts/{rollout_id}/step`
- `POST /rollouts/{rollout_id}/checkpoint`
- `POST /rollouts/{rollout_id}/restore`
- `POST /rollouts/{rollout_id}/simulate`
- `GET /rollouts/{rollout_id}/event_log`
- `GET /rollouts/{rollout_id}/render.svg`

HTTP smoke was previously checked for health, scenario, rollout, step, checkpoint, restore, simulate, and event-log flows.

## Policy Status

Primary policy for procedural 48×48 worlds: `policies/heuristic_max_achievements.py`.

Recorded sweep results (Rust lane, 100 seeds): see
`reports/policy_sweep/heuristic_max_achievements_v7.md`.

| Suite | Steps | Score (v7c) | Unique (v7c) |
|-------|-------|-------------|--------------|
| `policy_batch_default_v100.json` | 500 | 0.2274 | 32 |
| `policy_batch_default_v100_long.json` | 2000 | 0.2918 | 50 |

The code-policy sweep runner supports both lanes:

Python lane:

```bash
python3 tasks/craftax-singleplayer/scripts/run_policy_sweep.py \
  --suite tasks/craftax-singleplayer/defaults/policy_sweep/policy_smoke_v1.json \
  --output /tmp/craftax_python_policy_pickaxe.json
```

Rust lane:

```bash
python3 tasks/craftax-singleplayer/scripts/run_policy_sweep.py \
  --lane rust \
  --base-url http://127.0.0.1:8098 \
  --suite tasks/craftax-singleplayer/defaults/policy_sweep/policy_smoke_v1.json \
  --output /tmp/craftax_rust_policy_pickaxe.json
```

Last Rust HTTP smoke result:

```text
collect_wood: 5/5
place_table: 5/5
make_wood_pickaxe: 5/5
collect_stone: 5/5
make_stone_pickaxe: 2/5
score: 0.0818
mean_reward: 4.4
```

Last Rust procedural batch (heuristic_max_achievements v7c, 2026-06-21):

```text
500-step:  score=0.2274  unique=32  enter_dungeon=52%  place_furnace=96%
2000-step: score=0.2918  unique=50  enter_ice_realm=4%  eat_bat=18%  eat_snail=44%
reports: tasks/craftax-singleplayer/reports/policy_sweep/
```

Last Rust HTTP policy throughput:

```text
300 env steps / 0.592s = about 506.8 steps/sec
```

Important policy caveat:

- Rust policy mode passes `engine=None`, so policies cannot use Python `clone_for_sim()` lookahead.
- The improved baseline uses symbolic readout + ASCII routing instead, so it works in both lanes.
- Direct Rust bench throughput is much higher than HTTP policy throughput. HTTP policy throughput includes Python policy invocation, JSON serialization, and request overhead.

## Known Gaps / Next Work

1. Make the policy consistently get stone pickaxe, not only wood pickaxe.
   - Current smoke gets wood pickaxe 5/5 and stone pickaxe 2/5.
   - The remaining gap is policy routing/resource planning under dynamic spawned mobs and table placement layout, not engine parity.

2. Add a larger policy eval suite.
   - `policy_smoke_v1` is only 5 seeds x 60 steps.
   - A high-throughput policy benchmark should use more seeds and probably a longer horizon.

3. Add batch policy evaluation if HTTP throughput is not enough.
   - Current HTTP policy throughput: about 500 steps/sec.
   - Native direct engine bench: about 93k steps/sec.
   - For serious policy search, add an in-process Rust batch runner or a batched HTTP endpoint to avoid per-step HTTP overhead.

4. Keep Python as authority until any future fixture changes are regenerated from Python.
   - Rust should match Python fixtures, not redefine them.

5. If committing, review untracked scope carefully.
   - The entire Craftax task tree appears untracked in this checkout.
   - Do not accidentally include generated `target/`, `__pycache__/`, or `/tmp` reports.

## Cleanup Already Done

After the last policy/HTTP work:

- Rust service was stopped.
- `cargo clean --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml` was run.
- `__pycache__` directories under `tasks/craftax-singleplayer/` were removed.
- `gold_rust/target` was confirmed clean.

## Recommended Next Command Sequence

If picking this up fresh:

```bash
cd /Users/joshpurtell/Documents/GitHub/gamebench
cargo build --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml
python3 tasks/craftax-singleplayer/scripts/compare_gold_lanes.py --parity-only
cargo run --release --quiet --manifest-path tasks/craftax-singleplayer/gold_rust/Cargo.toml --bin craftax_gold -- --host 127.0.0.1 --port 8098
python3 tasks/craftax-singleplayer/scripts/run_policy_sweep.py --lane rust --base-url http://127.0.0.1:8098 --suite tasks/craftax-singleplayer/defaults/policy_sweep/policy_smoke_v1.json --output /tmp/craftax_rust_policy_pickaxe.json
```

Stop the service after the sweep and clean artifacts before handoff.
