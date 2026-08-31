# GameBench Craftax HTTP Contract

The Python and Rust gold lanes expose the same core routes. The local default
ports are `8097` for Python and `8098` for Rust.

- `GET /health`
- `POST /run_scenario`
- `POST /rollouts`
- `POST /rollouts/{rollout_id}/step`
- `GET /rollouts/{rollout_id}/readout`
- `GET /rollouts/{rollout_id}/event_log`
- `POST /rollouts/{rollout_id}/checkpoint`
- `POST /rollouts/{rollout_id}/restore`
- `POST /rollouts/{rollout_id}/simulate`
- `GET /rollouts/{rollout_id}/render.svg`
- `GET /rollouts/{rollout_id}/render.png` returns a current PNG frame. When
  `shared/assets/craftax/*.png` sprites are present (or
  `GAMEBENCH_CRAFTAX_ASSETS_DIR` points at them), frames use the Matthews/Craftax
  MA pixel art; otherwise lanes fall back to symbolic RGB tiles.
- `GET /rollouts/{rollout_id}/frames/manifest` returns captured PNG frame metadata.
- `GET /rollouts/{rollout_id}/frames/{step}.png` returns a captured PNG frame
  for a rollout step, or the current frame when requested on the current step.
- `GET /rollouts/{rollout_id}/replay.gif?through_step=N` returns a compact GIF
  assembled via ffmpeg palettegen/paletteuse. Rust automatic frame capture during
  reset/step requires replay mode (`GAMEBENCH_CRAFTAX_REPLAY_ENABLED=1` or
  `--replay`) plus task gates (`readouts.visual`, `stream.enabled`, or
  `stream.persist_frames`). On-demand PNG/GIF fetch captures the current frame
  even when those gates are off.

Checkpoint blobs use `gamebench.checkpoint.v1` and are intentionally lane-local.
They restore the resolved task, world grid, inventory, entity state, RNG state,
reward counters, and NEV cursor/events.
