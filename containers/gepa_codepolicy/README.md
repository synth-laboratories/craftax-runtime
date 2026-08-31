# GEPA Monty Python on Craftax `gold_rust`

Plane A (GEPA) is the orchestrator. Monty loads `choose_actions` in a child
process. Env is existing Rust gold HTTP. Optimizer never imports the candidate.

```bash
cd tasks/craftax-singleplayer
PYTHONPATH=.:gold_python:shared GAMEBENCH_MONTY_SANDBOX=process \
  python -m containers.gepa_codepolicy.stack
# GEPA: http://127.0.0.1:19300  env: :8098  policy: :19310

PYTHONPATH=.:gold_python:shared GAMEBENCH_MONTY_SANDBOX=process \
  python -m containers.gepa_codepolicy.try_policies
```

`POST /rollout` with `lever_bundle.values.policy_script` (`whole_file.v1` or
`unified_diff.v1`). `/program` advertises `load: monty_python` and
`gamebench.craftax_gold_rust.v1`.
