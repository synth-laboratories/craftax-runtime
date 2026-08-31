//! Newline-delimited JSON REPL for low-latency Craftax rollouts from Python workers.
//!
//! One engine session per process; no HTTP or cross-rollout locking.

use craftax_gamebench_gold::render::{
    encode_gif_via_ffmpeg, encode_png_rgb, frame_sha256, render_rgb_frame_from_world, RenderMode,
    RgbFrame, DEFAULT_RENDER_TILE_SIZE,
};
use craftax_gamebench_gold::CraftaxRustSession;
use base64::Engine;
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};

enum ReadoutMode {
    Full,
    Policy,
}

struct ReplState {
    session: Option<CraftaxRustSession>,
    replay: Option<ReplayCapture>,
}

struct ReplayCapture {
    frames: Vec<RgbFrame>,
    stride: usize,
    max_frames: usize,
    delay_cs: u16,
    steps_seen: usize,
}

fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut state = ReplState {
        session: None,
        replay: None,
    };
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(value) => value,
            Err(error) => {
                write_error(&mut stdout, Value::Null, error.to_string());
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let request = match serde_json::from_str::<Value>(&line) {
            Ok(value) => value,
            Err(error) => {
                write_error(&mut stdout, Value::Null, error.to_string());
                continue;
            }
        };
        let request_id = request.get("id").cloned().unwrap_or(Value::Null);
        let response = handle_request(&mut state, &request);
        write_response(&mut stdout, request_id, response);
    }
}

fn handle_request(state: &mut ReplState, request: &Value) -> Value {
    let op = request
        .get("op")
        .and_then(Value::as_str)
        .unwrap_or_default();
    match op {
        "ping" => json!({"ok": true, "lane": "rust_repl"}),
        "info" => json!({
            "ok": true,
            "action_names": craftax_gamebench_gold::action_names(),
            "glyph_legend": craftax_gamebench_gold::glyph_legend(),
        }),
        "close" => {
            state.session = None;
            state.replay = None;
            json!({"ok": true})
        }
        "reset" => {
            let task = request.get("task").cloned().unwrap_or_else(|| json!({}));
            let seed = request.get("seed").and_then(Value::as_i64);
            let readout_mode = match parse_readout_mode(request) {
                Ok(mode) => mode,
                Err(error) => return json!({"ok": false, "error": error}),
            };
            match CraftaxRustSession::reset_from_task(&task, seed) {
                Ok(session) => {
                    state.session = Some(session);
                    state.replay = parse_replay_capture(request);
                    capture_replay_frame(state, true);
                    session_payload(
                        state.session.as_ref().expect("session inserted"),
                        readout_mode,
                    )
                }
                Err(error) => json!({"ok": false, "error": error.to_string()}),
            }
        }
        "step" => {
            let Some(session) = state.session.as_mut() else {
                return json!({"ok": false, "error": "no active session; call reset first"});
            };
            let readout_mode = match parse_readout_mode(request) {
                Ok(mode) => mode,
                Err(error) => return json!({"ok": false, "error": error}),
            };
            let action = request
                .get("action")
                .cloned()
                .unwrap_or_else(|| json!("noop"));
            if !session.is_done() {
                if let Err(error) = session.step(&action) {
                    return json!({"ok": false, "error": error.to_string()});
                }
            }
            capture_replay_frame(state, false);
            let session = state.session.as_ref().expect("active session retained");
            session_payload(session, readout_mode)
        }
        "steps" => {
            if state.session.is_none() {
                return json!({"ok": false, "error": "no active session; call reset first"});
            }
            let readout_mode = match parse_readout_mode(request) {
                Ok(mode) => mode,
                Err(error) => return json!({"ok": false, "error": error}),
            };
            let Some(actions) = request.get("actions").and_then(Value::as_array) else {
                return json!({"ok": false, "error": "steps requires actions array"});
            };
            let mut steps_executed = 0usize;
            for action in actions {
                {
                    let session = state.session.as_mut().expect("active session retained");
                    if session.is_done() {
                        break;
                    }
                    if let Err(error) = session.step(action) {
                        return json!({"ok": false, "error": error.to_string()});
                    }
                }
                steps_executed += 1;
                capture_replay_frame(state, false);
            }
            let session = state.session.as_ref().expect("active session retained");
            let mut payload = session_payload(session, readout_mode);
            if let Some(object) = payload.as_object_mut() {
                object.insert("steps_executed".to_string(), json!(steps_executed));
            }
            payload
        }
        "readout" => {
            let Some(session) = state.session.as_ref() else {
                return json!({"ok": false, "error": "no active session; call reset first"});
            };
            let readout_mode = match parse_readout_mode(request) {
                Ok(mode) => mode,
                Err(error) => return json!({"ok": false, "error": error}),
            };
            session_payload(session, readout_mode)
        }
        "render_png" => render_png(state),
        "save_replay" => save_replay(state, request),
        other => json!({"ok": false, "error": format!("unknown op: {other}")}),
    }
}

fn render_png(state: &ReplState) -> Value {
    let Some(session) = state.session.as_ref() else {
        return json!({"ok": false, "error": "no active session; call reset first"});
    };
    let (width, height, rows) = render_rgb_frame_from_world(
        &session.world,
        DEFAULT_RENDER_TILE_SIZE,
        RenderMode::Auto,
    );
    let bytes = encode_png_rgb(width, height, &rows);
    json!({
        "ok": true,
        "content_type": "image/png",
        "width": width,
        "height": height,
        "sha256": frame_sha256(&bytes),
        "data_url": format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        ),
    })
}

fn parse_replay_capture(request: &Value) -> Option<ReplayCapture> {
    let replay = request.get("replay")?.as_object()?;
    if !replay.get("enabled").and_then(Value::as_bool).unwrap_or(false) {
        return None;
    }
    Some(ReplayCapture {
        frames: Vec::new(),
        stride: replay
            .get("stride")
            .and_then(Value::as_u64)
            .unwrap_or(100)
            .max(1) as usize,
        max_frames: replay
            .get("max_frames")
            .and_then(Value::as_u64)
            .unwrap_or(160)
            .clamp(2, 1000) as usize,
        delay_cs: replay
            .get("delay_cs")
            .and_then(Value::as_u64)
            .unwrap_or(8)
            .clamp(2, 100) as u16,
        steps_seen: 0,
    })
}

fn capture_replay_frame(state: &mut ReplState, force: bool) {
    let Some(capture) = state.replay.as_mut() else {
        return;
    };
    if !force {
        capture.steps_seen += 1;
    }
    if capture.frames.len() >= capture.max_frames
        || (!force && capture.steps_seen % capture.stride != 0)
    {
        return;
    }
    let Some(session) = state.session.as_ref() else {
        return;
    };
    capture.frames.push(render_rgb_frame_from_world(
        &session.world,
        DEFAULT_RENDER_TILE_SIZE,
        RenderMode::Auto,
    ));
}

fn save_replay(state: &mut ReplState, request: &Value) -> Value {
    let Some(session) = state.session.as_ref() else {
        return json!({"ok": false, "error": "no active session; call reset first"});
    };
    let Some(path) = request.get("path").and_then(Value::as_str) else {
        return json!({"ok": false, "error": "save_replay requires path"});
    };
    let Some(capture) = state.replay.as_mut() else {
        return json!({"ok": false, "error": "replay capture was not enabled at reset"});
    };
    if capture.frames.len() < capture.max_frames {
        capture.frames.push(render_rgb_frame_from_world(
            &session.world,
            DEFAULT_RENDER_TILE_SIZE,
            RenderMode::Auto,
        ));
    }
    let bytes = match encode_gif_via_ffmpeg(&capture.frames, capture.delay_cs) {
        Ok(bytes) => bytes,
        Err(error) => return json!({"ok": false, "error": error}),
    };
    if let Err(error) = std::fs::write(path, &bytes) {
        return json!({"ok": false, "error": error.to_string()});
    }
    let engine_sha256 = std::env::current_exe()
        .ok()
        .and_then(|engine_path| std::fs::read(engine_path).ok())
        .map(|engine_bytes| frame_sha256(&engine_bytes));
    json!({
        "ok": true,
        "path": path,
        "content_type": "image/gif",
        "sha256": frame_sha256(&bytes),
        "size_bytes": bytes.len(),
        "frame_count": capture.frames.len(),
        "steps_seen": capture.steps_seen,
        "engine_sha256": engine_sha256,
    })
}

fn parse_readout_mode(request: &Value) -> Result<ReadoutMode, String> {
    match request
        .get("readout_mode")
        .and_then(Value::as_str)
        .unwrap_or("full")
    {
        "full" => Ok(ReadoutMode::Full),
        "policy" => Ok(ReadoutMode::Policy),
        other => Err(format!("unsupported readout_mode: {other}")),
    }
}

fn session_payload(session: &CraftaxRustSession, readout_mode: ReadoutMode) -> Value {
    let readout = match readout_mode {
        ReadoutMode::Full => session.readout(),
        ReadoutMode::Policy => session.policy_readout(),
    };
    json!({
        "ok": true,
        "readout": readout,
        "terminated": session.private.get("terminated").and_then(Value::as_bool).unwrap_or(false),
        "truncated": session.private.get("truncated").and_then(Value::as_bool).unwrap_or(false),
        "reward": session.private.get("total_reward").cloned().unwrap_or(json!(0.0)),
    })
}

fn write_error(stdout: &mut io::Stdout, request_id: Value, message: String) {
    write_response(stdout, request_id, json!({"ok": false, "error": message}));
}

fn write_response(stdout: &mut io::Stdout, request_id: Value, mut response: Value) {
    if let Some(object) = response.as_object_mut() {
        object.insert("id".to_string(), request_id);
    }
    let encoded = serde_json::to_string(&response).unwrap_or_else(|error| {
        json!({"ok": false, "error": format!("encode response failed: {error}")}).to_string()
    });
    let _ = writeln!(stdout, "{encoded}");
    let _ = stdout.flush();
}
