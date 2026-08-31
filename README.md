# Craftax Single Player

Status: independent GameBench task with full symbolic Craftax parity surface.

This task implements symbolic Craftax gold lanes inside GameBench. It is based on
the Matthews JAX Craftax action and state vocabulary, but the runtime is local
GameBench code. It does not import the Crafter GameBench task or the external
JAX package at runtime.

Implemented surfaces:

- Python gold engine with deterministic world generation and authored initial states.
- Rust gold engine matching the Python gold fixtures across resolver output,
  legacy NEV strings, and grid hashes for all 36 gold scenarios.
- Full Matthews Craftax full-mode vocabulary for 42 actions, 37 block types,
  5 item types, canonical mobs/projectiles, and 66 achievements.
- Symbolic implementations for every canonical action and achievement family,
  including advanced floors, chests, books, spells, potions, armor,
  enchantments, named mobs, bow fire, and necromancer completion.
- Structured NEV events, legacy event strings, and fixture verification.
- JSON checkpoints with restore/clone/simulate support.
- Symbolic readouts, observation text, ASCII/SVG render state, and Python/Rust
  HTTP services.
- Code-policy baseline, sweep runner, and policy contract tests.

Cybernetic example: sparse achievement-target planner over symbolic Craftax — [docs/cybernetic_example.md](docs/cybernetic_example.md). Full exotic-cybernetics handoff: [exotic_cybernetics/HANDOFF.md](exotic_cybernetics/HANDOFF.md).

Quick commands:

```bash
cd tasks/craftax-singleplayer
PYTHONPATH=gold_python:shared python scripts/generate_eventlogs.py
PYTHONPATH=gold_python:shared python scripts/verify_gold_nev.py
PYTHONPATH=gold_python:shared python scripts/verify_parity_surface.py
PYTHONPATH=gold_python:shared python scripts/verify_restore_equivalence.py
PYTHONPATH=gold_python:shared python scripts/verify_import_boundary.py
PYTHONPATH=gold_python:shared python scripts/bench_checkpoint.py --iterations 50
PYTHONPATH=gold_python:shared python scripts/run_policy_tests.py
cargo build --manifest-path gold_rust/Cargo.toml
cargo run --quiet --manifest-path gold_rust/Cargo.toml --bin restore_check
python3 scripts/compare_gold_lanes.py --resolver-only
python3 scripts/compare_gold_lanes.py --parity-only
python3 scripts/compare_gold_lanes.py --parity-only --perf
python3 scripts/run_service.py --lane python --port 8097
python3 scripts/run_service.py --lane rust --port 8098
```

The task is not a JAX wrapper and does not attempt pixel-level reproduction of
the Matthews renderer or stochastic JAX world generator. Parity here means the
GameBench symbolic runtime covers the full Craftax action/state/achievement
surface without importing JAX or Crafter code.
