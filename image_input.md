# Image observations for Craftax agents

**Date:** 2026-08-13
**Policy under test:** Luna medium ReAct — `openai/gpt-5.6-luna` via OpenRouter,
`reasoning_effort: medium`, conversational loop with fixed-threshold compaction
**World:** `craftax_default` (48×48, 9 levels), requested 1000 steps / 150 turns,
10 seeds. The service silently clamped `max_llm_turns` to 128 at the time of this
experiment, so the 127–128-turn survivors (894–1000 steps) were turn-capped, not
purely step-capped. The clamp is now 1024; the historical results below are unchanged.

## Headline

Pixels alone are not better than symbols. **Pixels *plus* symbols are a different
regime**, and they unlock a phase of the game neither modality reaches alone.

| | mean | median | max | achievements | survivors >500 steps | tokens |
|---|---:|---:|---:|---:|---:|---:|
| text | 7.09 | 4.55 | 16.00 | 18 | 3/10 | 4.0M |
| image | 7.37 | 7.00 | 15.60 | 19 | 4/10 | 3.7M |
| **both** | **12.90** | **14.50** | **22.10** | **25** | **7/10** | 7.6M |

`image` vs `text` is +0.28 — inside noise (see *Statistical floor* below).
`both` vs `text` is **+5.81**, more than double the measured noise band.

### What only `both` reached

`enter_dungeon` · `open_chest` · `find_bow` · `collect_sapphire` ·
`defeat_orc_mage` · `drink_potion` · `place_torch`

These are all dungeon achievements. **The agent descended below the surface for the
first time in any configuration tested.** Every other run — toy board, stateless,
conversation, scratchpad, text-only, image-only — topped out on the surface tech
tree at stone tools and a furnace. This is a capability that was absent and is now
present, which is far stronger evidence than a shifted mean.

Only `image` had that `text` lacked: `collect_drink`. Nothing was unique to `text`.

### Why they are complementary

The frame carries spatial structure: where the stone face is, which way water runs,
where a zombie stands relative to the player. The symbolic text carries exact
scalars: `wood=38`, `health=6`, which achievements are banked.

Each is bad at the other's job. You cannot read `food=9` off pixels, and you cannot
navigate off a comma-separated tile list. Neither alone was enough; together the
agent survives long enough to find a ladder down.

The two are also confounded by survival: `both` lived longer, and survival drives
the mean. "Better perception → better decisions" and "better perception → survives →
more time to accumulate" are probably the same mechanism, but this data does not
decompose them.

## The observation the env renders

`sprites::render_observation_frame(world, view_radius, tile_size)`.

- **9×9 FOV window**, the same window `local_map` reports — identical radius and
  identical block → item → entity → projectile → player draw precedence.
- **Out-of-bounds renders as darkness**, not a wrapped or clamped tile.
- **Vitals and inventory HUD** beneath the map: `health/food/drink/energy/mana`,
  then non-zero `wood/stone/coal/iron/diamond/ruby/sapphire/sapling/torches/arrows/
  books`, tiered tools (`pickaxe` level 1–4 → wood/stone/iron/diamond sprite), bow,
  and attributes. Counts drawn from the `0.png`–`9.png` digit sprites.

Inspect it directly: `GET /rollouts/:id/observation.png`. That endpoint exists
because the first version of this renderer had no HUD at all and I only caught it
by looking.

### Config

| key | default | |
|---|---|---|
| `observation_mode` | `text` | `text` · `image` · `both` |
| `render_tile_size` | 32 | px per tile |
| `view_radius` | 4 | 4 → 9×9 window |
| `keep_recent_frames` | 2 | older frames drop their payload |

## Three mistakes that would have produced a wrong answer

**1. The renderer drew the entire 48×48 world.** `render_rgb_frame_from_world` is a
debugging view. Sending it to a policy leaks every unexplored tile and silently
breaks partial observability — image mode would have "won" by seeing the whole map.

**2. The first cropped version had no HUD.** Terrain only: no health, no inventory.
That is not a modality swap, it is a modality swap *plus* an information loss, so
the comparison would have measured missing facts rather than perception. The HUD is
what makes `text` and `image` carry the same information.

**3. Frames accumulated and timed out, biasing the sample.** Images are re-sent
every turn until compaction; one `both` seed burned **1,066,008 tokens**. The client
had a 120s timeout, so oversized multimodal requests failed and surfaced as
`502 Bad Gateway` — easy to misread as provider flakiness. It is worse than random
loss: **the seeds that failed were the deep survivors**, because those are the ones
whose transcripts grow large enough to time out. The surviving sample skewed toward
early deaths, which is exactly backwards.

Fixed by pruning to the most recent 2 frames (older ones keep their text, lose the
payload) and raising the timeout to 600s. Result: prompt tokens grow 828 → 4,444
over 20 turns instead of reaching 500k–1M, and the rerun lost **zero** seeds where
the previous attempt lost seven.

Compaction also strips frames before summarizing — serializing the raw transcript
would ship every accumulated data URI to the summarizer, and it cannot use pixels in
a text brief anyway.

## Statistical floor

Three independent runs of an **identical** configuration produced means of
**9.61 / 9.51 / 7.09**. The mean is dominated by how many of ten seeds survive the
early game — survivors score ~15, deaths ~4 — and that is a coin flip. A 5/10 vs
3/10 swing moves the mean ~2.5 points on its own.

Consequences:

- **Anything under ~5 points of mean difference is unreadable at n=10.** The
  scratchpad ablation's +1.88 is noise, not a weak positive.
- **Achievement union and survival count are the robust metrics.** A mean drifts;
  "this achievement was never reached" does not.
- `both` clears the bar on both counts — +5.81 mean and a categorical capability
  gain — which is why it is reported as a result rather than a trend.

For a number rather than a direction, run 30+ seeds per arm with survival count as
the primary metric.

## Known fidelity gap

This engine has **no line-of-sight model**. `local_map` is a plain bounded window;
the only light modelling is `calculate_light_level` for day/night. The black regions
in reference Craftax screenshots come from its visibility system, which does not
exist here — our frame shows every tile in the 9×9 box. Closing that needs a
visibility model in the engine, not a renderer change.

Achievements have no representation on the reference screen either, so in
`observation_mode: image` they arrive as a single `achievements:` text line rather
than an invented icon row.

## Fixes so this class of failure cannot recur

The frame-accumulation bug was not really about frames. It was two general
defects that happened to meet: a transient failure that was neither retried nor
distinguished from a permanent one, and a summary that averaged whatever seeds
happened to survive.

**Loud, classified, retried failures** — `call_policy_lm` in `craftax_gold.rs`.
Every failure used to collapse onto one opaque `policy_http` 502, which is why a
local timeout on an oversized request read as provider flakiness. Now:

- timeouts, transport errors, decode failures, 429 and 5xx are named separately,
  and the message carries elapsed time, attempt number and **request size in
  bytes** — the fact that would have identified the real cause immediately;
- transient classes retry with exponential backoff (`policy_lm_attempts`,
  default 3);
- a 4xx that is not 429 fails immediately as `policy_request_rejected`, because
  that is our bug and retrying a malformed request only wastes money.

**A driver that refuses to average a biased sample** —
`scripts/run_craftax_eval.py`. Losing a seed is not neutral: long rollouts fail
preferentially, so dropping them biases an arm toward early deaths, and biases
each arm *differently* depending on how expensive its turns are. The driver
retries per seed, then **exits non-zero and refuses to print a comparison** when
any arm is short, unless `--allow-missing` is passed explicitly. It also refuses
when any seed reports `is_reference_world: false`, so a toy-board score can never
again be presented as a Craftax result.

It reports survival count and achievement union as the headline, with the mean
carrying a percentile bootstrap interval — bootstrap rather than a normal
approximation, because Craftax rewards are bimodal (early deaths vs deep
survivors) and not remotely bell-shaped. The footer states the n-dependent floor
in the output itself rather than leaving it to whoever reads the table.

```
       arm   scored  survivors  achv    mean           95% CI  median
      text    2/2        0/2       4    3.50       [3.0, 4.0]    3.50
      both    2/2        0/2       3    2.00       [1.0, 3.0]    2.00
```
