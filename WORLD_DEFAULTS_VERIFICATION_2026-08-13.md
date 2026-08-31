# Verification — real Craftax vs the toy board

Companion to `WORLD_DEFAULTS_FINDING_2026-08-13.md`. Same policy, same 10 seeds,
same harness. The only change is the world.

**Policy:** Luna medium ReAct — `openai/gpt-5.6-luna` via OpenRouter,
`reasoning_effort: medium`, 4–8 actions per call.

## Result

| | A — `policy_dev_small` 16×16 | B — `craftax_default` 48×48 |
|---|---:|---:|
| mean reward | 0.98 | **5.06** |
| median | 0.60 | **4.80** |
| min / max | −1.00 / 4.00 | 1.00 / **11.00** |
| achievement union | 8 | **12** |

**Paired: 9 seeds better, 1 worse, 0 tied. Mean delta +4.08.**

| seed | A | B | Δ |
|---:|---:|---:|---:|
| 0 | 4.00 | 8.00 | +4.00 |
| 1 | 3.00 | 5.00 | +2.00 |
| 2 | 2.80 | 2.40 | −0.40 |
| 3 | 1.10 | 8.00 | +6.90 |
| 4 | −1.00 | 1.00 | +2.00 |
| 5 | 1.60 | 6.60 | +5.00 |
| 6 | −0.90 | 11.00 | +11.90 |
| 7 | −1.00 | 4.60 | +5.60 |
| 8 | 0.10 | 2.00 | +1.90 |
| 9 | 0.10 | 2.00 | +1.90 |

## The achievements matter more than the mean

Present on real Craftax, **absent from all ten toy-board seeds**:

`collect_stone` · `collect_coal` · `make_wood_pickaxe` · `make_wood_sword` ·
`place_furnace` · `defeat_skeleton`

The early tech tree was not *hard* on the 16×16 board — wood → table → pickaxe →
stone was outside a 120-step budget on a map with 16% of the trees. It was
unreachable. Two achievements go the other way (`collect_drink`, `place_plant`):
the toy board leaves an agent with little else to do.

The old receipt's mean of 0.98 was therefore not a Craftax capability measurement.
It was a measurement of a test fixture.

## Cost

237,967 tokens across 10 seeds; 29 min wall at 5-way parallelism. The service does
not record `cost_usd`, so no dollar figure is claimed here — token counts are the
metered quantity.

## Reproduce

```bash
cargo run --release --manifest-path gold_rust/Cargo.toml --bin craftax_gold -- \
  --host 127.0.0.1 --port 8110

curl -X POST http://127.0.0.1:8110/rollout -H 'content-type: application/json' -d '{
  "seed": 0,
  "env": {"seed": 0, "config": {"max_steps": 200, "world": {"use_default": "craftax_default"}}},
  "policy": {"config": {"use_lm": true, "provider": "openrouter",
    "model": "openai/gpt-5.6-luna", "reasoning_effort": "medium",
    "max_llm_turns": 30, "min_actions_per_call": 4, "max_actions_per_call": 8}}}'
```

Check `summary.world.is_reference_world` on the response. If it is not `true`, the
number is not a Craftax result.

---

## Follow-on: the ReAct loop was also wrong

Fixing the world exposed the next constraint. The harness was not a ReAct loop at
all — every call was a fresh `[system, user]` pair, and the only state carried
between turns was `last_actions[-16:]`. On a real episode that meant **152 of 168
actions were dropped with no summary**: the agent could not remember terrain it had
seen twenty actions earlier. Prompt size was flat at ~475 tokens/call, so it would
run forever without ever needing to compact, because nothing accumulated.

Replaced with a conversational loop: `system → user(task) → assistant(tool_calls) →
tool(observation) → …`, the environment exposed as an `interact` tool, and automatic
compaction at a fixed context threshold that replaces the middle of the transcript
with a model-written summary while preserving the system message and the opening
task verbatim.

### Third configuration, same 10 seeds

| | A toy/stateless | B real/stateless | C real/conversation |
|---|---:|---:|---:|
| steps · turns | 200 · 30 | 200 · 30 | 1000 · 150 |
| mean | 0.98 | 5.06 | **9.61** |
| median | 0.60 | 4.80 | **10.05** |
| max | 4.00 | 11.00 | **17.00** |
| achievement union | 8 | 12 | **19** |

C vs B paired: 8 better, 2 worse. Mean +4.55.

Achievements reached only in C: `collect_iron`, `make_stone_pickaxe`,
`make_stone_sword`, `place_stone`, `make_arrow`, `collect_drink`, `place_plant` —
two full tech tiers past anything the previous setup produced.

### The result is bimodal

```
survived past 500 steps:  5/10 — mean 15.00
died before 500 steps  :  5/10 — mean  4.22
```

Almost nothing in between. Either a zombie finds the agent in the first ~50 steps, or
it survives and the loop compounds. All five survivors finished at 894–1000 steps and
127–128 turns — **step-capped, not turn-capped**, so a larger budget would have gone
further. Deaths are the environment working as intended, not a harness fault.

Loop health: 756/764 turns (99%) used real tool calls; 23 compactions, all at the
fixed threshold, none producing an orphaned tool result. 5.78M prompt + 105k
completion tokens, ~$0.64, 48 min wall at 10-way parallelism.

### Caveat on attribution

C changes two variables against B — conversation memory *and* 5× the budget — so the
+4.55 cannot be split between them from this data. C also ran on a different binary
than the later ablation arms, so it is a reference point, not a control.
