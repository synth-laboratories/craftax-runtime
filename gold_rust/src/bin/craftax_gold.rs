use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::extract::ws::{Message, WebSocket};
use axum::http::{header, StatusCode};
use axum::response::{sse::{Event, KeepAlive, Sse}, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use craftax_gamebench_gold::render::{
    encode_gif_via_ffmpeg, encode_png_rgb, frame_sha256, render_rgb_frame_from_world,
    RenderMode, RgbFrame, DEFAULT_RENDER_TILE_SIZE,
};
use craftax_gamebench_gold::{run_entry, CraftaxRustSession, EventRecord};
use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::convert::Infallible;
use tokio::sync::{broadcast, OwnedSemaphorePermit, Semaphore};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone, Copy)]
struct ReplaySettings {
    enabled: bool,
}

/// One rollout's state, independently lockable.
type SessionHandle = Arc<Mutex<RolloutSession>>;

#[derive(Clone)]
struct AppState {
    // A handle per session, not a session per map. The map lock now only
    // guards lookup; the work happens under the session's own lock, so two
    // rollouts stepping at once no longer serialise against each other.
    sessions: Arc<Mutex<HashMap<String, SessionHandle>>>,
    telemetry: Arc<Mutex<HashMap<String, broadcast::Sender<Value>>>>,
    telemetry_history: Arc<Mutex<HashMap<String, Vec<Value>>>>,
    session_slots: Arc<Semaphore>,
    max_sessions: usize,
    replay: ReplaySettings,
}

const DEFAULT_MAX_SESSIONS: usize = 128;

const DEFAULT_TASK: &str = r#"{
  "schema": "gamebench.task.craftax.v1",
  "task_id": "manual",
  "scenario_id": "manual",
  "world": {"use_default": "craftax_default"},
  "rules": {"base": "symbolic_no_homeostasis"},
  "readouts": {"profile": "symbolic_compact"}
}"#;

struct RolloutSession {
    // Held for exactly as long as the server retains this session. Dropping the
    // session via DELETE returns admission capacity without a second counter.
    _admission_permit: OwnedSemaphorePermit,
    engine: CraftaxRustSession,
    started_at: Instant,
    checkpoints: HashMap<String, Vec<u8>>,
    frames: HashMap<u64, Vec<u8>>,
    gif_frames: HashMap<u64, RgbFrame>,
    /// GELO/GEPA terminal record when an optimizer rollout has finished.
    terminal_record: Option<Value>,
    status: String,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
}

struct TraceCapture {
    requested: String,
    required: bool,
    control_url: Option<String>,
    control_token: Option<String>,
    client: Option<reqwest::Client>,
    capture_id: Option<String>,
    opened: Option<Value>,
    failure: Option<String>,
    uploaded_digests: HashSet<String>,
}

impl Drop for TraceCapture {
    fn drop(&mut self) {
        let Some(capture_id) = self.capture_id.take() else {
            return;
        };
        let Some(url) = self.control_url.clone() else {
            return;
        };
        let Some(client) = self.client.clone() else {
            return;
        };
        let token = self.control_token.clone();
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let mut request = client
                    .post(format!("{url}/captures/{capture_id}/seal"))
                    .json(&json!({
                        "status": "interrupted",
                        "termination": {"reason": "craftax_rollout_aborted"},
                    }));
                if let Some(token) = token {
                    request = request.bearer_auth(token);
                }
                let _ = request.send().await;
            });
        }
    }
}

impl TraceCapture {
    fn off() -> Self {
        Self {
            requested: "off".into(),
            required: false,
            control_url: None,
            control_token: None,
            client: None,
            capture_id: None,
            opened: None,
            failure: None,
            uploaded_digests: HashSet::new(),
        }
    }

    async fn open(
        policy_cfg: &Value,
        rollout_id: &str,
        client: reqwest::Client,
    ) -> Result<Self, ApiError> {
        let requested = policy_cfg
            .get("capture")
            .and_then(Value::as_str)
            .unwrap_or("off")
            .to_string();
        if !matches!(requested.as_str(), "off" | "best_effort" | "required") {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                code: "invalid_capture_mode",
                message: "policy.config.capture must be off, best_effort, or required".into(),
            });
        }
        let required = requested == "required";
        let control_url = policy_cfg
            .get("capture_url")
            .and_then(Value::as_str)
            .map(str::to_string)
            .or_else(|| std::env::var("SYNTH_TRACE_CONTROL_URL").ok())
            .map(|url| url.trim_end_matches('/').to_string());
        let mut capture = Self {
            requested: requested.clone(),
            required,
            control_url,
            control_token: std::env::var("SYNTH_TRACE_CONTROL_TOKEN").ok(),
            client: Some(client),
            capture_id: None,
            opened: None,
            failure: None,
            uploaded_digests: HashSet::new(),
        };
        if requested == "off" {
            return Ok(capture);
        }
        let Some(url) = capture.control_url.clone() else {
            capture.fail("SYNTH_TRACE_CONTROL_URL/capture_url is not configured".into())?;
            return Ok(capture);
        };
        let mut request = capture.client.as_ref().expect("capture client").post(format!("{url}/captures")).json(&json!({
            "rollout_id": rollout_id,
            "labels": {"workload": "craftax-singleplayer", "emitter": "gold_rust"},
            "capture_mode": requested,
        }));
        if let Some(token) = capture.control_token.as_deref() {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => match response.json::<Value>().await {
                Ok(opened) => {
                    capture.capture_id = opened
                        .get("capture_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    capture.opened = Some(opened);
                    if capture.capture_id.is_none() {
                        capture.fail("capture open response omitted capture_id".into())?;
                    }
                }
                Err(err) => capture.fail(format!("capture open response: {err}"))?,
            },
            Ok(response) => capture.fail(format!("capture open returned HTTP {}", response.status()))?,
            Err(err) => capture.fail(format!("capture open failed: {err}"))?,
        }
        Ok(capture)
    }

    fn fail(&mut self, message: String) -> Result<(), ApiError> {
        self.failure = Some(message.clone());
        if self.required {
            Err(ApiError {
                status: StatusCode::FAILED_DEPENDENCY,
                code: "trace_capture_required",
                message,
            })
        } else {
            Ok(())
        }
    }

    async fn emit_turn(&mut self, mut payload: Value) -> Result<(), ApiError> {
        let Some(capture_id) = self.capture_id.clone() else {
            return Ok(());
        };
        if let Some(messages) = payload.get("messages").cloned() {
            match self.externalize_images(messages).await {
                Ok(messages) => payload["messages"] = messages,
                Err(err) => {
                    if self.interrupt_capture(&capture_id, &err).await {
                        self.capture_id = None;
                    }
                    return self.fail(err);
                }
            }
        }
        let Some(url) = self.control_url.as_deref() else {
            return self.fail("capture control URL disappeared".into());
        };
        let mut request = self
            .client
            .as_ref()
            .expect("open capture has an HTTP client")
            .post(format!("{url}/captures/{capture_id}/events"))
            .json(&json!({"event_type": "craftax.turn", "payload": payload}));
        if let Some(token) = self.control_token.as_deref() {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => {
                let message = format!("capture event returned HTTP {}", response.status());
                if self.interrupt_capture(&capture_id, &message).await {
                    self.capture_id = None;
                }
                self.fail(message)
            }
            Err(err) => {
                let message = format!("capture event failed: {err}");
                if self.interrupt_capture(&capture_id, &message).await {
                    self.capture_id = None;
                }
                self.fail(message)
            }
        }
    }

    async fn externalize_images(&mut self, mut messages: Value) -> Result<Value, String> {
        let Some(items) = messages.as_array_mut() else {
            return Ok(messages);
        };
        for message_index in 0..items.len() {
            let part_count = items[message_index]
                .get("content")
                .and_then(Value::as_array)
                .map(Vec::len)
                .unwrap_or(0);
            for part_index in 0..part_count {
                let data_url = items[message_index]
                    .pointer(&format!("/content/{part_index}/image_url/url"))
                    .and_then(Value::as_str)
                    .map(str::to_string);
                let Some(data_url) = data_url else { continue };
                let Some(encoded) = data_url.strip_prefix("data:image/png;base64,") else {
                    continue;
                };
                let content = BASE64.decode(encoded).map_err(|err| format!("invalid trace frame: {err}"))?;
                let digest = format!("sha256:{:x}", Sha256::digest(&content));
                if !self.uploaded_digests.contains(&digest) {
                    self.upload_image(&content, message_index, &digest).await?;
                    self.uploaded_digests.insert(digest.clone());
                }
                items[message_index]["content"][part_index]["image_url"] = json!({
                    "digest": digest,
                    "media_type": "image/png",
                });
            }
        }
        Ok(messages)
    }

    async fn upload_image(
        &self,
        content: &[u8],
        message_index: usize,
        digest: &str,
    ) -> Result<(), String> {
        let capture_id = self.capture_id.as_deref().ok_or("capture is not open")?;
        let url = self.control_url.as_deref().ok_or("capture control URL is missing")?;
        let mut request = self
            .client
            .as_ref()
            .expect("open capture has an HTTP client")
            .post(format!("{url}/captures/{capture_id}/artifacts"))
            .json(&json!({
                "role": "observation",
                "media_type": "image/png",
                "logical_name": format!("craftax-frame-{message_index}-{digest}.png"),
                "content_base64": BASE64.encode(content),
                "visibility": "private",
            }));
        if let Some(token) = self.control_token.as_deref() {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => Ok(()),
            Ok(response) => Err(format!("capture artifact returned HTTP {}", response.status())),
            Err(err) => Err(format!("capture artifact failed: {err}")),
        }
    }

    async fn interrupt_capture(&self, capture_id: &str, reason: &str) -> bool {
        let Some(url) = self.control_url.as_deref() else { return false };
        let mut request = self
            .client
            .as_ref()
            .expect("open capture has an HTTP client")
            .post(format!("{url}/captures/{capture_id}/seal"))
            .json(&json!({
                "status": "interrupted",
                "termination": {"reason": "craftax_emission_failed", "detail": reason},
            }));
        if let Some(token) = self.control_token.as_deref() {
            request = request.bearer_auth(token);
        }
        matches!(request.send().await, Ok(response) if response.status().is_success())
    }

    async fn seal(&mut self) -> Result<Value, ApiError> {
        let Some(capture_id) = self.capture_id.clone() else {
            return Ok(if self.requested == "off" {
                json!({"capture": "off", "status": "disabled"})
            } else {
                json!({
                    "capture": self.requested,
                    "status": "unavailable",
                    "reason": self.failure,
                })
            });
        };
        let Some(url) = self.control_url.as_deref() else {
            self.fail("capture control URL disappeared before seal".into())?;
            return Ok(json!({"capture": self.requested, "status": "unavailable"}));
        };
        let mut request = self
            .client
            .as_ref()
            .expect("open capture has an HTTP client")
            .post(format!("{url}/captures/{capture_id}/seal"))
            .json(&json!({"status": "completed"}));
        if let Some(token) = self.control_token.as_deref() {
            request = request.bearer_auth(token);
        }
        match request.send().await {
            Ok(response) if response.status().is_success() => {
                let mut sealed = response.json::<Value>().await.map_err(|err| ApiError {
                    status: StatusCode::BAD_GATEWAY,
                    code: "trace_capture_seal",
                    message: err.to_string(),
                })?;
                self.capture_id = None;
                sealed["capture"] = json!(self.requested);
                Ok(sealed)
            }
            Ok(response) => {
                self.fail(format!("capture seal returned HTTP {}", response.status()))?;
                Ok(json!({"capture": self.requested, "status": "unavailable", "reason": self.failure}))
            }
            Err(err) => {
                self.fail(format!("capture seal failed: {err}"))?;
                Ok(json!({"capture": self.requested, "status": "unavailable", "reason": self.failure}))
            }
        }
    }
}

fn trace_safe_message(mut message: Value) -> Value {
    // OpenRouter may return an opaque provider-signed reasoning_details.data blob.
    // It is not exposed thinking, is not useful for replay/training, grows every
    // later request snapshot, and random URL-safe bytes can trip literal secret
    // scanners (including an incidental `sk-` substring). Capture the assistant
    // content as thinking and omit this transport-only continuation token.
    if let Some(object) = message.as_object_mut() {
        object.remove("reasoning_details");
    }
    message
}

fn trace_safe_messages(messages: Value) -> Value {
    match messages {
        Value::Array(items) => Value::Array(items.into_iter().map(trace_safe_message).collect()),
        other => other,
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(json!({
                "error": {
                    "code": self.code,
                    "message": self.message
                }
            })),
        )
            .into_response()
    }
}

#[derive(Deserialize)]
struct ScenarioRequest {
    task: Value,
}

#[derive(Deserialize)]
struct RolloutRequest {
    task: Option<Value>,
    seed: Option<i64>,
    telemetry: Option<Value>,
}

#[derive(Deserialize)]
struct StepRequest {
    action: Value,
}

#[derive(Deserialize)]
struct RestoreRequest {
    blob: String,
}

#[derive(Deserialize)]
struct SimulateRequest {
    blob: String,
    sequences: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct ReplayGifQuery {
    through_step: Option<u64>,
}

#[derive(Deserialize)]
struct TaskInfoQuery {
    task_id: Option<String>,
}

#[tokio::main]
async fn main() {
    let mut host = "127.0.0.1".to_string();
    let mut port = 8098_u16;
    let mut replay_enabled = replay_enabled_from_env();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--host" => host = args.next().unwrap_or(host),
            "--port" => {
                port = args
                    .next()
                    .and_then(|value| value.parse().ok())
                    .unwrap_or(port)
            }
            "--replay" => replay_enabled = true,
            "--no-replay" => replay_enabled = false,
            _ => {}
        }
    }

    let max_sessions = configured_max_sessions();
    let state = AppState {
        sessions: Arc::new(Mutex::new(HashMap::new())),
        telemetry: Arc::new(Mutex::new(HashMap::new())),
        telemetry_history: Arc::new(Mutex::new(HashMap::new())),
        session_slots: Arc::new(Semaphore::new(max_sessions)),
        max_sessions,
        replay: ReplaySettings {
            enabled: replay_enabled,
        },
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/task_catalog", get(task_catalog))
        .route("/task_info", get(task_info))
        .route("/run_scenario", post(run_scenario_route))
        // GELO/GEPA optimizers post GELO-shaped bodies to /rollout; GameBench
        // scenario runners also use /rollout with `{task:...}`. Dispatch both.
        .route("/rollout", post(optimizer_or_scenario_rollout))
        .route("/rollouts", post(create_rollout))
        .route("/reset", post(create_rollout))
        .route("/rollouts/:rollout_id", get(get_rollout_record).delete(delete_rollout))
        .route("/rollouts/:rollout_id/step", post(step))
        .route("/rollouts/:rollout_id/readout", get(readout))
        .route("/rollouts/:rollout_id/state", get(rollout_state))
        .route("/rollouts/:rollout_id/terminate", post(terminate_rollout))
        .route("/dataset", get(gepa_dataset))
        .route("/dataset/rows", post(gepa_dataset_rows))
        .route("/program", get(gepa_program))
        .route("/metadata", get(optimizer_metadata))
        .route("/compatibility", get(compatibility))
        .route("/rollouts/:rollout_id/event_log", get(event_log))
        .route("/rollouts/:rollout_id/events", get(event_log))
        .route("/rollouts/:rollout_id/stream", get(rollout_stream))
        .route("/rollouts/:rollout_id/ws", get(rollout_ws))
        .route("/rollouts/:rollout_id/checkpoint", post(checkpoint))
        .route("/rollouts/:rollout_id/checkpoints", post(checkpoint))
        .route("/rollouts/:rollout_id/resume", post(resume_optimizer_rollout))
        .route("/rollouts/:rollout_id/restore", post(restore))
        .route("/rollouts/:rollout_id/simulate", post(simulate))
        .route("/rollouts/:rollout_id/render.svg", get(render_svg))
        .route("/rollouts/:rollout_id/render.png", get(render_png_route))
        // What the agent actually sees: FOV window + vitals/inventory HUD.
        .route("/rollouts/:rollout_id/observation.png", get(observation_png_route))
        .route("/rollouts/:rollout_id/frames/manifest", get(frame_manifest_route))
        .route("/rollouts/:rollout_id/frames/:step", get(frame_png_route))
        .route("/rollouts/:rollout_id/replay.gif", get(replay_gif_route))
        .layer(CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}"))
        .await
        .expect("bind Craftax Rust gold service");
    axum::serve(listener, app)
        .await
        .expect("serve Craftax Rust gold service");
}

async fn health(State(app): State<AppState>) -> Json<Value> {
    let available_sessions = app.session_slots.available_permits();
    Json(json!({
        "ok": true,
        "lane": "rust",
        "env_family": "craftax-singleplayer",
        "sessions": app.sessions.lock().unwrap().len(),
        "session_capacity": app.max_sessions,
        "session_capacity_available": available_sessions,
        "session_capacity_in_use": app.max_sessions.saturating_sub(available_sessions),
        "replay_enabled": app.replay.enabled,
    }))
}

fn configured_max_sessions() -> usize {
    match std::env::var("SYNTH_CRAFTAX_MAX_SESSIONS") {
        Ok(raw) => raw.parse::<usize>().ok().filter(|value| *value > 0).unwrap_or_else(|| {
            panic!("SYNTH_CRAFTAX_MAX_SESSIONS must be a positive integer, got {raw:?}")
        }),
        Err(std::env::VarError::NotPresent) => DEFAULT_MAX_SESSIONS,
        Err(err) => panic!("could not read SYNTH_CRAFTAX_MAX_SESSIONS: {err}"),
    }
}

fn reserve_session_slot(app: &AppState) -> Result<OwnedSemaphorePermit, ApiError> {
    app.session_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| ApiError {
            status: StatusCode::TOO_MANY_REQUESTS,
            code: "too_many_sessions",
            message: format!(
                "Craftax session capacity is exhausted (max {}); delete completed rollouts and retry",
                app.max_sessions
            ),
        })
}

async fn info(State(app): State<AppState>) -> Json<Value> {
    let capabilities = vec![
        "rollout",
        "checkpoint",
        "nev_log",
        "symbolic_readout",
        "render_svg",
        "render_png",
        "frames_manifest",
        "frame_png",
        "replay_gif",
        "simulate_from_checkpoint",
        "task_catalog",
        "task_info",
        "rollout_stream_sse",
        "rollout_stream_ws",
    ];
    Json(json!({
        "env_family": "craftax-singleplayer",
        "lane": "rust",
        "capabilities": capabilities,
        "streaming": {
            "schema": "synth.rollout.event.v1",
            "transports": ["sse", "websocket"],
            "supports_resume": false,
            "frame_formats": ["png"]
        },
        "replay_enabled": app.replay.enabled,
        "action_names": craftax_gamebench_gold::action_names(),
        "glyph_legend": craftax_gamebench_gold::glyph_legend(),
    }))
}

async fn task_catalog() -> Json<Value> {
    Json(task_catalog_value())
}

fn task_catalog_value() -> Value {
	let instances = (1001_i64..=1032)
		.map(|seed| ("train", seed))
		.chain((2001_i64..=2008).map(|seed| ("test", seed)))
		.map(|(split, seed)| json!({
			"task_instance_id": format!("craftax:{split}:{seed}"),
			"task_id": "manual",
			"split": split,
			"tags": ["craftax", split, "craftax_default"],
			"metadata": {
				"seed": seed,
				"world_profile": "craftax_default",
				"rules": "symbolic_no_homeostasis",
				"readout_profile": "symbolic_compact"
			},
			"rollout": {
				"seed": seed,
				"task": {
					"schema": "gamebench.task.craftax.v1",
					"task_id": "manual",
					"scenario_id": format!("seed-{seed}"),
					"world": {"use_default": "craftax_default", "seed": seed},
					"rules": {"base": "symbolic_no_homeostasis"},
					"readouts": {"profile": "symbolic_compact"}
				}
			}
		}))
		.collect::<Vec<_>>();
	json!({
        "schema": "synth.container.task_catalog.v1",
        "tasks": [{
            "task_id": "manual",
            "name": "Craftax single-player",
            "description": "Explore, gather, craft, fight, and survive in a symbolic Craftax world.",
            "family": "craftax-singleplayer",
            "scenario_id": "manual",
            "default": true
        }],
		"instances": instances,
		"metadata": {
			"instance_count": 40,
			"seed_policy": "curated_fixed_v1",
			"filterable_fields": ["split", "metadata.seed", "metadata.world_profile", "metadata.rules"]
		}
	})
}

async fn task_info(Query(query): Query<TaskInfoQuery>) -> Result<Json<Value>, ApiError> {
    let task_id = query.task_id.as_deref().unwrap_or("manual");
    if task_id != "manual" {
        return Err(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "task_not_found",
            message: format!("unknown task_id: {task_id}"),
        });
    }
    Ok(Json(json!({
        "schema": "synth.container.task_info.v1",
        "task": {
            "task_id": "manual",
            "name": "Craftax single-player",
            "description": "Explore, gather resources, craft equipment, fight creatures, and survive in a symbolic Craftax world.",
            "family": "craftax-singleplayer",
            "scenario_id": "manual"
        },
        "objective": "Select valid actions that make progress through the Craftax technology tree while keeping the player alive.",
        "output_space": {
            "kind": "interactive_action",
            "contract": "Choose exactly one action name per environment step.",
            "valid_actions": craftax_gamebench_gold::action_names()
        },
        "metrics": {
            "primary": "achievements_unlocked",
            "secondary": ["survival", "inventory_progress", "depth_reached"]
        },
        "constraints": {
            "state_observation": "symbolic",
            "default_world": "craftax_default",
            "rules": "symbolic_no_homeostasis"
        },
        "metadata": {
            "readout_profile": "symbolic_compact",
            "supports_seed": true,
            "supports_checkpointing": true
        }
    })))
}

async fn run_scenario_route(Json(body): Json<ScenarioRequest>) -> Result<Json<Value>, ApiError> {
    Ok(Json(run_entry(&body.task).map_err(bad_request)?))
}

fn is_optimizer_rollout_body(body: &Value) -> bool {
    body.get("arms").is_some()
        || body.get("candidate").is_some()
        || body.get("schema_version")
            .and_then(Value::as_str)
            .is_some_and(|s| s.contains("goex") || s.contains("gepa") || s.contains("rollout_request"))
        || body.get("submission_mode").is_some()
        || body.get("policy").is_some()
}


/// Render the agent's current view as a base64 PNG, cropped to the same window
/// the symbolic observation shows.
///
/// `render_rgb_frame_from_world` draws the whole world. Handing that to a model
/// would leak every unexplored tile and quietly break partial observability, so
/// the frame is cropped to (player ± view_radius) — the same 9x9 window
/// `observation_text` describes. Text and image modes then carry the same
/// information in different modalities, which is the only way the comparison
/// means anything.
fn observation_png_b64(
    world: &craftax_gamebench_gold::CraftaxWorld,
    readout: &Value,
    view_radius: i64,
    tile: u32,
) -> String {
    use base64::Engine as _;
    let _ = readout;
    let (w, h, rows) =
        craftax_gamebench_gold::render_observation_frame(world, view_radius, tile);
    let png = encode_png_rgb(w, h, &rows);
    base64::engine::general_purpose::STANDARD.encode(png)
}


/// Call the policy LM with bounded retries, and fail loudly about *why*.
///
/// The previous code mapped every failure — connect error, read timeout, 429,
/// 5xx, malformed body — onto one opaque `policy_http` 502. That made a local
/// bug (an oversized multimodal request exceeding the client timeout) look
/// exactly like provider flakiness, and because long transcripts are what time
/// out, the resulting seed losses were biased toward deep survivors rather than
/// random. A biased silent drop is far worse than a loud stop.
async fn call_policy_lm(
    client: &reqwest::Client,
    url: &str,
    api_key: &str,
    body: &Value,
    attempts: usize,
) -> Result<Value, ApiError> {
    let request_bytes = serde_json::to_vec(body).map(|v| v.len()).unwrap_or(0);
    let mut last: String = "no attempt made".into();
    for attempt in 1..=attempts.max(1) {
        let started = Instant::now();
        let response = client
            .post(url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await;
        match response {
            Err(err) => {
                // Transient by nature: connect failures and read timeouts.
                let kind = if err.is_timeout() { "timeout" } else { "transport" };
                last = format!(
                    "{kind} after {:.1}s on attempt {attempt}/{attempts}                      (request {request_bytes} bytes): {err}",
                    started.elapsed().as_secs_f64()
                );
            }
            Ok(response) => {
                let status = response.status();
                let payload = response.json::<Value>().await;
                match payload {
                    Err(err) => {
                        last = format!(
                            "decode failed on attempt {attempt}/{attempts} (HTTP {}): {err}",
                            status.as_u16()
                        );
                    }
                    Ok(payload) => {
                        if status.is_success() {
                            return Ok(payload);
                        }
                        let retryable = status.as_u16() == 429 || status.is_server_error();
                        last = format!(
                            "HTTP {} on attempt {attempt}/{attempts} (request {request_bytes}                              bytes): {}",
                            status.as_u16(),
                            serde_json::to_string(&payload).unwrap_or_default().chars().take(400).collect::<String>()
                        );
                        // A 4xx that is not 429 is our bug, not theirs. Retrying
                        // an malformed request just wastes time and money.
                        if !retryable {
                            return Err(ApiError {
                                status: StatusCode::BAD_REQUEST,
                                code: "policy_request_rejected",
                                message: last,
                            });
                        }
                    }
                }
            }
        }
        if attempt < attempts.max(1) {
            let backoff = std::time::Duration::from_millis(500 * (1 << (attempt - 1).min(5)));
            tokio::time::sleep(backoff).await;
        }
    }
    Err(ApiError {
        status: StatusCode::BAD_GATEWAY,
        code: "policy_lm_unavailable",
        message: format!("policy LM failed after {attempts} attempts; last: {last}"),
    })
}


/// Where rollout event logs are spooled. Override with GAMEBENCH_CRAFTAX_NEV_DIR.
fn nev_dir() -> PathBuf {
    std::env::var("GAMEBENCH_CRAFTAX_NEV_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            // task_dir() is crate-private; derive the same location from the
            // manifest so the spool sits with the task, not the cwd.
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .map(|p| p.join(".out/nev"))
                .unwrap_or_else(|| PathBuf::from(".out/nev"))
        })
}

/// Write a rollout's NEV log to disk and describe where it went.
///
/// Returns a reference even on failure, with `error` set — losing evidence
/// silently is how a run ends up unauditable, so a spool failure is reported in
/// the rollout record rather than swallowed.
fn spool_nev(rollout_id: &str, events: &[EventRecord]) -> Value {
    let dir = nev_dir();
    let path = dir.join(format!("{rollout_id}.jsonl"));
    let mut body = String::new();
    for event in events {
        match serde_json::to_string(event) {
            Ok(line) => {
                body.push_str(&line);
                body.push('\n');
            }
            Err(err) => {
                return json!({
                    "count": events.len(),
                    "error": format!("serialize failed: {err}"),
                });
            }
        }
    }
    let digest = format!("sha256:{:x}", Sha256::digest(body.as_bytes()));
    if let Err(err) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(&path, &body)) {
        return json!({
            "count": events.len(),
            "digest": digest,
            "error": format!("write {} failed: {err}", path.display()),
        });
    }
    json!({
        "count": events.len(),
        "digest": digest,
        "path": path.to_string_lossy(),
        "url": format!("/rollouts/{rollout_id}/events"),
        "bytes": body.len(),
    })
}

async fn optimizer_or_scenario_rollout(
    State(app): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    if is_optimizer_rollout_body(&body) {
        return run_optimizer_rollout(app, body).await;
    }
    if let Some(task) = body.get("task") {
        return Ok(Json(run_entry(task).map_err(bad_request)?));
    }
    // Fall through to interactive create_rollout shape.
    let seed = body.get("seed").and_then(Value::as_i64);
    let task = body.get("task").cloned().unwrap_or_else(default_task);
    create_rollout_from_parts(app, task, seed, body.get("telemetry").cloned()).await
}

async fn create_rollout(
    State(app): State<AppState>,
    Json(body): Json<RolloutRequest>,
) -> Result<Json<Value>, ApiError> {
    let task = body.task.unwrap_or_else(default_task);
    create_rollout_from_parts(app, task, body.seed, body.telemetry).await
}

async fn create_rollout_from_parts(
    app: AppState,
    task: Value,
    seed: Option<i64>,
    telemetry: Option<Value>,
) -> Result<Json<Value>, ApiError> {
    let admission_permit = reserve_session_slot(&app)?;
    let engine = CraftaxRustSession::reset_from_task(&task, seed).map_err(bad_request)?;
    let rollout_id = Uuid::new_v4().to_string();
    let mut session = RolloutSession {
        _admission_permit: admission_permit,
        engine,
        started_at: Instant::now(),
        checkpoints: HashMap::new(),
        frames: HashMap::new(),
        gif_frames: HashMap::new(),
        terminal_record: None,
        status: "running".into(),
    };
    capture_frame_if_enabled(&mut session, app.replay, false);
    let mut payload = rollout_payload(&rollout_id, &session);
    if telemetry
        .as_ref()
        .and_then(|value| value.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        capture_frame_if_enabled(&mut session, app.replay, true);
        let (sender, _) = broadcast::channel(256);
        app.telemetry.lock().unwrap().insert(rollout_id.clone(), sender.clone());
        payload["stream"] = json!({
            "schema": "synth.rollout.stream.v1",
            "sse_url": format!("/rollouts/{rollout_id}/stream"),
            "websocket_url": format!("/rollouts/{rollout_id}/ws"),
            "cursor": 0
        });
        let event = rollout_event(&rollout_id, &payload);
        app.telemetry_history.lock().unwrap().insert(rollout_id.clone(), vec![event.clone()]);
        let _ = sender.send(event);
    }
    app.sessions.lock().unwrap().insert(rollout_id, Arc::new(Mutex::new(session)));
    Ok(Json(payload))
}

async fn run_optimizer_rollout(app: AppState, body: Value) -> Result<Json<Value>, ApiError> {
    let admission_permit = reserve_session_slot(&app)?;
    let seed = body
        .pointer("/env/seed")
        .or_else(|| body.get("seed"))
        .and_then(Value::as_i64)
        .unwrap_or(101);
    let policy_cfg = body
        .pointer("/policy/config")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let max_steps = body
        .pointer("/env/config/max_steps")
        .or_else(|| policy_cfg.get("max_steps"))
        .or_else(|| body.get("segment_steps"))
        .and_then(Value::as_u64)
        .unwrap_or(32)
        // The reference world runs to 5000 steps. A 512 ceiling silently
        // truncated every real-world run; the default stays small so no
        // existing caller changes behaviour by accident.
        .clamp(1, 5000) as usize;
    let max_llm_turns = policy_cfg
        .get("max_llm_turns")
        .or_else(|| body.pointer("/go_ex/max_llm_turns"))
        .and_then(Value::as_u64)
        .unwrap_or(16)
        .clamp(1, 1024) as usize;
    let use_lm = policy_cfg
        .get("use_lm")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let rollout_id = body
        .get("rollout_id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| Uuid::new_v4().to_string());
    let mut trace_capture = if policy_cfg
        .get("capture")
        .and_then(Value::as_str)
        .unwrap_or("off")
        == "off"
    {
        TraceCapture::off()
    } else {
        let trace_http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .map_err(|err| ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "trace_http_client",
                message: err.to_string(),
            })?;
        TraceCapture::open(&policy_cfg, &rollout_id, trace_http_client).await?
    };
    // The world used to be hardcoded to `policy_dev_small` here: a 16x16, 3-level
    // board with 16% of vanilla tree density and a 120-step budget. Every
    // optimizer, GEPA, GELO and SFT rollout went through this function, so all
    // of them scored a toy arena and reported it as a Craftax result — the
    // early tech tree (wood -> table -> pickaxe -> stone) is not reachable
    // there. The world is now caller-configurable and defaults to the reference
    // world; a caller that wants the fast dev board has to ask for it by name.
    let mut world = body
        .pointer("/env/config/world")
        .or_else(|| body.pointer("/env/world"))
        .or_else(|| body.pointer("/task/world"))
        .or_else(|| policy_cfg.get("world"))
        .cloned()
        .unwrap_or_else(|| json!({"use_default": "craftax_default"}));
    if let Some(obj) = world.as_object_mut() {
        obj.insert("seed".into(), json!(seed));
    }
    let world_preset = world
        .get("use_default")
        .and_then(Value::as_str)
        .unwrap_or("inline")
        .to_string();
    // A score is only comparable to a real Craftax run when the board is the
    // reference board and no resource density has been scaled.
    let scaled_resources: Vec<String> = world
        .get("densities")
        .and_then(Value::as_object)
        .map(|o| o.keys().cloned().collect())
        .unwrap_or_default();
    let is_reference_world = world_preset == "craftax_default" && scaled_resources.is_empty();
    let mut task = default_task();
    if let Some(obj) = task.as_object_mut() {
        obj.insert("world".into(), world.clone());
        obj.insert("scenario_id".into(), json!(format!("seed-{seed}")));
    }
    let mut engine = CraftaxRustSession::reset_from_task(&task, Some(seed)).map_err(bad_request)?;
    let actions = craftax_gamebench_gold::action_names();
    let valid: Vec<String> = actions.iter().map(|s| (*s).to_string()).collect();

    let system_prompt = body
        .pointer("/policy/config/system_prompt")
        .or_else(|| body.pointer("/policy/config/react_system_prompt"))
        .or_else(|| body.pointer("/candidate/react_system_prompt"))
        .or_else(|| body.pointer("/arms/0/react_system_prompt"))
        .and_then(Value::as_str)
        .unwrap_or(
            "You are playing GameBench Craftax. Reply with JSON only, for example \
             {\"actions\":[\"do\",\"right\",\"do\"]}. Prioritize survival, wood, stone, tools, \
             and achievement progress.",
        )
        .to_string();
    let provider = body
        .pointer("/policy/config/provider")
        .or_else(|| body.pointer("/policy/provider"))
        .and_then(Value::as_str)
        .unwrap_or("groq")
        .to_ascii_lowercase();
    let model = body
        .pointer("/policy/config/model")
        .or_else(|| body.pointer("/policy/model"))
        .and_then(Value::as_str)
        .unwrap_or(match provider.as_str() {
            "deepseek" => "deepseek-v4-flash",
            "openai" => "gpt-4.1-mini",
            "gemini" => "gemini-3.1-flash-lite",
            _ => "openai/gpt-oss-120b",
        })
        .to_string();
    let credential_mode = body
        .pointer("/policy/credential_mode")
        .or_else(|| body.pointer("/policy/config/credential_mode"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_ascii_lowercase();
    let workshop_proxy = matches!(credential_mode.as_str(), "workshop_proxy" | "proxy");
    let inference_url = if workshop_proxy {
        let raw = body
            .pointer("/policy/inference_url")
            .or_else(|| body.pointer("/policy/config/inference_url"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                bad_request("inference_url is required when credential_mode=workshop_proxy")
            })?;
        let lower = raw.to_ascii_lowercase();
        if lower.contains("api.openai.com")
            || lower.contains("127.0.0.1")
            || lower.contains("localhost")
        {
            return Err(bad_request(
                "workshop_proxy inference_url must be the Workshop container proxy",
            ));
        }
        if raw.ends_with("/chat/completions") {
            raw.to_string()
        } else {
            format!("{}/chat/completions", raw.trim_end_matches('/'))
        }
    } else {
        body.pointer("/policy/config/inference_url")
            .or_else(|| body.pointer("/policy/inference_url"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| default_inference_url(&provider))
    };
    // Reasoning effort is part of the policy identity — "Luna medium" is a
    // different policy from "Luna high" — so it has to be configurable and
    // recorded, not inferred from the provider name.
    let reasoning_effort = body
        .pointer("/policy/config/reasoning_effort")
        .or_else(|| body.pointer("/policy/reasoning_effort"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let api_key_env = body
        .pointer("/policy/config/api_key_env")
        .or_else(|| body.pointer("/policy/api_key_env"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| default_api_key_env(&provider));
    let api_key = if workshop_proxy {
        std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "workshop-proxy".to_string())
    } else {
        std::env::var(&api_key_env)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .or_else(|| std::env::var("GROQ_API_KEY").ok().filter(|s| !s.trim().is_empty()))
            .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok().filter(|s| !s.trim().is_empty()))
            .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|s| !s.trim().is_empty()))
            .or_else(|| std::env::var("GEMINI_API_KEY").ok().filter(|s| !s.trim().is_empty()))
            .unwrap_or_default()
    };
    let temperature = policy_cfg
        .get("temperature")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let max_tokens = body
        .pointer("/policy/config/max_tokens")
        .or_else(|| body.pointer("/policy/max_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(512)
        .clamp(64, 4096) as u32;
    let min_actions = policy_cfg
        .get("min_actions_per_call")
        .and_then(Value::as_u64)
        .unwrap_or(3)
        .clamp(1, 32) as usize;
    let max_actions = policy_cfg
        .get("max_actions_per_call")
        .and_then(Value::as_u64)
        .unwrap_or(12)
        .clamp(1, 32) as usize;
    let objective = "Make measurable Craftax achievement progress while staying alive.".to_string();

    // ReAct shape. `conversation` is the classic loop: an accumulating
    // system/user/assistant-tool_call/tool-result transcript that is compacted
    // by a model-written summary when the context crosses a threshold.
    // `stateless` is the older single-turn prompt kept for A/B comparison — it
    // carries only a 16-action tail window and forgets everything else.
    let react_mode = policy_cfg
        .get("react_mode")
        .and_then(Value::as_str)
        .unwrap_or("conversation")
        .to_string();
    let context_token_budget = policy_cfg
        .get("context_token_budget")
        .and_then(Value::as_u64)
        .unwrap_or(16000)
        .clamp(2000, 400_000) as u64;
    let compact_at = policy_cfg
        .get("compact_at")
        .and_then(Value::as_f64)
        .unwrap_or(0.7)
        .clamp(0.1, 0.95);
    let compact_threshold = (context_token_budget as f64 * compact_at) as u64;
    let keep_recent_messages = policy_cfg
        .get("keep_recent_messages")
        .and_then(Value::as_u64)
        .unwrap_or(8)
        .clamp(2, 64) as usize;
    // Optional durable scratchpad. The compaction summary is model-written and
    // lossy; a scratchpad is agent-authored and pinned, so a plan survives
    // verbatim no matter how many times the middle is summarized away.
    // How the agent perceives the world: symbolic text, a rendered frame, or
    // both. The frame is cropped to the same view window as the text.
    let observation_mode = policy_cfg
        .get("observation_mode")
        .and_then(Value::as_str)
        .unwrap_or("text")
        .to_string();
    let render_tile_size = policy_cfg
        .get("render_tile_size")
        .and_then(Value::as_u64)
        .unwrap_or(32)
        .clamp(8, 64) as u32;
    let view_radius = policy_cfg
        .get("view_radius")
        .and_then(Value::as_i64)
        .unwrap_or(4)
        .clamp(1, 24);
    // Frames are re-sent on every turn until compaction, so an unbounded
    // transcript costs ~1M tokens per episode and eventually exceeds the
    // request timeout. Only the most recent frames carry usable information;
    // older ones become a placeholder.
    // Retries for transient upstream failures. A lost seed is not neutral: the
    // rollouts that fail are the long ones, so silent drops bias a comparison
    // toward early deaths.
    // Events are the only durable record this path produces; default to
    // returning them. Opt out for throughput runs that do not need evidence.
    let include_events = policy_cfg
        .get("include_events")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let policy_lm_attempts = policy_cfg
        .get("policy_lm_attempts")
        .and_then(Value::as_u64)
        .unwrap_or(3)
        .clamp(1, 8) as usize;
    let keep_recent_frames = policy_cfg
        .get("keep_recent_frames")
        .and_then(Value::as_u64)
        .unwrap_or(2)
        .clamp(1, 16) as usize;
    let send_image = observation_mode == "image" || observation_mode == "both";
    let send_text = observation_mode != "image";
    let scratchpad_enabled = policy_cfg
        .get("enable_scratchpad")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut scratchpad = String::new();
    let mut scratchpad_writes: usize = 0;
    let mut invalid_actions_total: usize = 0;
    let mut compactions: Vec<Value> = Vec::new();
    let mut last_prompt_tokens: u64 = 0;

    let mut action_history: Vec<String> = Vec::new();
    let mut turns: Vec<Value> = Vec::new();
    let mut usage_total = json!({"prompt_tokens": 0u64, "completion_tokens": 0u64, "total_tokens": 0u64});
    let mut llm_calls: usize = 0;
    let mut policy_label = "craftax_react_lm".to_string();

    if use_lm {
        if api_key.is_empty() {
            return Err(ApiError {
                status: StatusCode::BAD_REQUEST,
                code: "missing_api_key",
                message: format!(
                    "Craftax LM rollout requires {api_key_env} (provider={provider})"
                ),
            });
        }
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .map_err(|err| ApiError {
                status: StatusCode::INTERNAL_SERVER_ERROR,
                code: "http_client",
                message: err.to_string(),
            })?;
        if react_mode == "conversation" {
            // ── Classic ReAct: one accumulating transcript ──────────────────
            // system + user(task) + assistant(tool_calls) + tool(result) + ...
            // The environment is a tool the model calls; observations come back
            // as tool results. Nothing is re-templated per turn, so the model
            // sees the whole episode until compaction summarizes the middle.
            let tools = json!([{
                "type": "function",
                "function": {
                    "name": "interact",
                    "description": "Act in the world: execute Craftax actions in order, then return the resulting observation. Plan a short coherent batch.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {"type": "string", "enum": valid.clone()},
                                "description": "Actions to execute sequentially."
                            }
                        },
                        "required": ["actions"],
                        "additionalProperties": false
                    }
                }
            }]);
            // Compaction is infrastructure, not a policy decision: it fires
            // automatically at a fixed context threshold. Exposing it as a tool
            // makes context management part of what is under test, which is a
            // different experiment.
            let tools = if scratchpad_enabled {
                let mut t = tools.as_array().cloned().unwrap_or_default();
                t.push(json!({
                    "type": "function",
                    "function": {
                        "name": "scratchpad",
                        "description": "Record or revise your long-term plan and map knowledge. This text is pinned to your context and survives context compaction verbatim, so put anything here you must not forget: where resources are, what the current goal is, what has already been tried. Rewrite it in full each time.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The complete new scratchpad contents, replacing whatever was there."}
                            },
                            "required": ["content"],
                            "additionalProperties": false
                        }
                    }
                }));
                json!(t)
            } else {
                tools
            };

            let opening = format!(
                "{objective}\n\nYou are playing Craftax. Call the `interact` tool to act; you will get the new observation back as the tool result. Plan {min_actions}-{max_actions} coherent actions per call.\n\nvalid_actions={}\n\nInitial observation:\n{}",
                valid.join(", "),
                engine
                    .readout()
                    .get("observation_text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
            );
            let opening_content = if send_image {
                let mut parts = vec![json!({"type": "text", "text": opening})];
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {"url": format!(
                        "data:image/png;base64,{}",
                        observation_png_b64(&engine.world, &engine.readout(), view_radius, render_tile_size)
                    )}
                }));
                json!(parts)
            } else {
                json!(opening)
            };
            let mut messages: Vec<Value> = vec![
                json!({"role": "system", "content": system_prompt}),
                json!({"role": "user", "content": opening_content}),
            ];
            // Index 2 is the pinned scratchpad when enabled. Compaction keeps
            // the head intact, so this slot is never summarized away.
            let head_len = if scratchpad_enabled { 3 } else { 2 };
            if scratchpad_enabled {
                messages.push(json!({
                    "role": "user",
                    "content": "[scratchpad] empty — call the `scratchpad` tool to record your plan."
                }));
            }

            while !engine.is_done()
                && (engine
                    .private
                    .get("step_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize)
                    < max_steps
                && llm_calls < max_llm_turns
            {
                let steps_done = engine
                    .private
                    .get("step_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                let steps_remaining = max_steps.saturating_sub(steps_done);
                let llm_remaining = max_llm_turns.saturating_sub(llm_calls);

                let mut request_body = json!({
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                });
                if let Some(effort) = reasoning_effort.as_deref() {
                    request_body.as_object_mut().unwrap()
                        .insert("reasoning_effort".into(), json!(effort));
                }
                let trace_messages = trace_safe_messages(request_body
                    .get("messages")
                    .cloned()
                    .unwrap_or_else(|| json!([])));
                let trace_steps_before = steps_done;
                let trace_reward_before = engine
                    .private
                    .get("total_reward")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let trace_achievements_before = engine
                    .private
                    .get("achievements")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                let payload = call_policy_lm(
                    &client,
                    &inference_url,
                    &api_key,
                    &request_body,
                    policy_lm_attempts,
                ).await?;
                accumulate_usage(&mut usage_total, payload.get("usage"));
                last_prompt_tokens = payload
                    .pointer("/usage/prompt_tokens")
                    .and_then(Value::as_u64)
                    .unwrap_or(last_prompt_tokens);

                let msg = payload.pointer("/choices/0/message").cloned().unwrap_or(json!({}));
                let assistant_text = extract_assistant_text(&payload);
                let tool_calls = msg.get("tool_calls").and_then(Value::as_array).cloned();
                let trace_tool_calls = tool_calls.clone().unwrap_or_default();

                // Keep the assistant turn verbatim so tool_call_ids stay valid.
                let transcript_len_before_response = messages.len();
                messages.push(msg.clone());

                let mut executed: Vec<String> = Vec::new();
                let mut planned: Vec<String> = Vec::new();
                let mut turn_invalid: usize = 0;
                let turn_observation = engine
                    .readout()
                    .get("observation_text")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let used_tool_call = tool_calls.as_ref().is_some_and(|c| !c.is_empty());
                match tool_calls {
                    Some(calls) if !calls.is_empty() => {
                        for call in calls {
                            let args_raw = call
                                .pointer("/function/arguments")
                                .and_then(Value::as_str)
                                .unwrap_or("{}");
                            let fn_name = call
                                .pointer("/function/name")
                                .and_then(Value::as_str)
                                .unwrap_or("step");
                            if fn_name == "scratchpad" {
                                let content = serde_json::from_str::<Value>(args_raw)
                                    .ok()
                                    .and_then(|v| v.get("content").and_then(Value::as_str).map(str::to_string))
                                    .unwrap_or_default();
                                if !content.trim().is_empty() {
                                    scratchpad = content;
                                    scratchpad_writes += 1;
                                    if messages.len() > 2 {
                                        messages[2] = json!({
                                            "role": "user",
                                            "content": format!("[scratchpad — pinned, survives compaction]\n{scratchpad}")
                                        });
                                    }
                                }
                                messages.push(json!({
                                    "role": "tool",
                                    "tool_call_id": call.get("id").and_then(Value::as_str).unwrap_or(""),
                                    "content": format!("scratchpad saved ({} chars). It is pinned to your context.", scratchpad.len())
                                }));
                                continue;
                            }
                            let batch_cap = max_actions.min(steps_remaining).max(1);
                            let (batch, invalid) = parse_actions_batch_counted(args_raw, &valid, batch_cap);
                            invalid_actions_total += invalid;
                            turn_invalid += invalid;
                            planned.extend(batch.iter().cloned());
                            for action in &batch {
                                if engine.is_done() {
                                    break;
                                }
                                let now = engine.private.get("step_index")
                                    .and_then(Value::as_u64).unwrap_or(0) as usize;
                                if now >= max_steps {
                                    break;
                                }
                                engine.step(&json!(action)).map_err(|err| ApiError {
                                    status: StatusCode::BAD_REQUEST,
                                    code: "step_failed",
                                    message: err.to_string(),
                                })?;
                                executed.push(action.clone());
                                action_history.push(action.clone());
                            }
                            let readout = engine.readout();
                            let obs = readout.get("observation_text")
                                .and_then(Value::as_str).unwrap_or("");
                            // A `tool` message must be a plain string, so the
                            // frame cannot ride along here. It follows as its
                            // own user turn.
                            let result = format!(
                                "executed={}\nsteps_remaining={}\nllm_calls_remaining={}{}",
                                serde_json::to_string(&executed).unwrap_or_else(|_| "[]".into()),
                                max_steps.saturating_sub(
                                    engine.private.get("step_index")
                                        .and_then(Value::as_u64).unwrap_or(0) as usize),
                                llm_remaining.saturating_sub(1),
                                if send_text {
                                    format!("\n\n{obs}")
                                } else {
                                    // Visual-only. The frame carries terrain,
                                    // vitals and inventory, but the reference
                                    // Craftax screen has no achievement row —
                                    // so achievements are the one fact that
                                    // must come through as text, or the agent
                                    // cannot see its own progress at all.
                                    let unlocked = engine
                                        .private
                                        .get("achievements")
                                        .and_then(Value::as_array)
                                        .map(|a| {
                                            a.iter()
                                                .filter_map(Value::as_str)
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        })
                                        .unwrap_or_default();
                                    format!(
                                        "\n\nachievements: {}",
                                        if unlocked.is_empty() { "none yet" } else { &unlocked }
                                    )
                                }
                            );
                            messages.push(json!({
                                "role": "tool",
                                "tool_call_id": call.get("id").and_then(Value::as_str).unwrap_or(""),
                                "content": result
                            }));
                            if send_image {
                                messages.push(json!({
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": "Current view:"},
                                        {"type": "image_url", "image_url": {"url": format!(
                                            "data:image/png;base64,{}",
                                            observation_png_b64(&engine.world, &engine.readout(), view_radius, render_tile_size)
                                        )}}
                                    ]
                                }));
                            }
                        }
                    }
                    _ => {
                        // No tool call. Accept a plain JSON action batch so a
                        // model that ignores the tool schema still advances,
                        // and tell it plainly what it should have done.
                        let batch_cap = max_actions.min(steps_remaining).max(1);
                        let (batch, invalid) = parse_actions_batch_counted(&assistant_text, &valid, batch_cap);
                        invalid_actions_total += invalid;
                        turn_invalid += invalid;
                        planned.extend(batch.iter().cloned());
                        for action in &batch {
                            if engine.is_done() {
                                break;
                            }
                            let now = engine.private.get("step_index")
                                .and_then(Value::as_u64).unwrap_or(0) as usize;
                            if now >= max_steps {
                                break;
                            }
                            engine.step(&json!(action)).map_err(|err| ApiError {
                                status: StatusCode::BAD_REQUEST,
                                code: "step_failed",
                                message: err.to_string(),
                            })?;
                            executed.push(action.clone());
                            action_history.push(action.clone());
                        }
                        let readout = engine.readout();
                        let obs = readout.get("observation_text")
                            .and_then(Value::as_str).unwrap_or("");
                        messages.push(json!({
                            "role": "user",
                            "content": format!(
                                "Use the `interact` tool next time. executed={}\n\n{}",
                                serde_json::to_string(&executed).unwrap_or_else(|_| "[]".into()),
                                obs
                            )
                        }));
                    }
                }

                // Keep only the most recent frames as pixels; older image
                // messages keep their text and lose the payload.
                if send_image {
                    let mut seen = 0usize;
                    for message in messages.iter_mut().rev() {
                        let has_image = message
                            .get("content")
                            .and_then(Value::as_array)
                            .is_some_and(|parts| {
                                parts.iter().any(|p| {
                                    p.get("type").and_then(Value::as_str) == Some("image_url")
                                })
                            });
                        if !has_image {
                            continue;
                        }
                        seen += 1;
                        if seen > keep_recent_frames {
                            let text = message
                                .get("content")
                                .and_then(Value::as_array)
                                .map(|parts| {
                                    parts
                                        .iter()
                                        .filter_map(|p| p.get("text").and_then(Value::as_str))
                                        .collect::<Vec<_>>()
                                        .join(" ")
                                })
                                .unwrap_or_default();
                            message["content"] =
                                json!(format!("{text} [frame dropped — superseded]"));
                        }
                    }
                }

                llm_calls += 1;
                let trace_tool_results: Vec<Value> = messages
                    .iter()
                    .skip(transcript_len_before_response + 1)
                    .filter(|message| {
                        message.get("role").and_then(Value::as_str) == Some("tool")
                    })
                    .cloned()
                    .collect();
                let trace_steps_after = engine
                    .private
                    .get("step_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let trace_reward_after = engine
                    .private
                    .get("total_reward")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.0);
                let trace_achievements_after = engine
                    .private
                    .get("achievements")
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                turns.push(json!({
                    "llm_call": llm_calls,
                    "assistant": assistant_text,
                    "actions": executed,
                    "planned": planned,
                    // The observation this turn conditioned on. Without it a
                    // turn cannot become an SFT example — curation rejects
                    // "no observation to condition on" — and in image mode the
                    // symbolic text is not in the tool result either.
                    "observation_text": turn_observation,
                    "invalid_actions": turn_invalid,
                    // The exact target an SFT example must reproduce. In
                    // tool-call mode `assistant` carries the model's thinking
                    // and the action lives in the tool call, so the target has
                    // to be stated separately or every turn looks empty.
                    "target": serde_json::to_string(&json!({"actions": planned}))
                        .unwrap_or_else(|_| "{\"actions\":[]}".into()),
                    "prompt_tokens": last_prompt_tokens,
                    "message_count": messages.len(),
                    // Distinguishes a real tool call from the text fallback.
                    // Without it you cannot tell whether the loop is actually
                    // running on tool calls or quietly limping on JSON parsing.
                    "used_tool_call": used_tool_call,
                }));
                trace_capture
                    .emit_turn(json!({
                        "llm_call": llm_calls,
                        "messages": trace_messages,
                        "assistant_message": trace_safe_message(msg),
                        "thinking": assistant_text,
                        "tool_calls": trace_tool_calls,
                        "tool_results": trace_tool_results,
                        "usage": payload.get("usage").cloned().unwrap_or_else(|| json!({})),
                        "env_transition": {
                            "steps_before": trace_steps_before,
                            "steps_after": trace_steps_after,
                            "actions": executed,
                            "reward_before": trace_reward_before,
                            "reward_after": trace_reward_after,
                            "reward_delta": trace_reward_after - trace_reward_before,
                            "achievements_before": trace_achievements_before,
                            "achievements_after": trace_achievements_after,
                        },
                        "compactions": compactions,
                    }))
                    .await?;

                // ── Compaction ─────────────────────────────────────────────
                // Fixed context window: compact when the real prompt token
                // count crosses the threshold. Not agent-decided.
                if last_prompt_tokens > compact_threshold
                    && messages.len() > keep_recent_messages + head_len
                {
                    // Never split an assistant(tool_calls)/tool(result) pair:
                    // an orphaned tool message is a hard API error.
                    let mut start = messages.len().saturating_sub(keep_recent_messages);
                    while start > head_len
                        && messages[start].get("role").and_then(Value::as_str) == Some("tool")
                    {
                        start -= 1;
                    }
                    if start > head_len {
                        let middle: Vec<Value> = messages[head_len..start].to_vec();
                        // Strip base64 frames before summarizing. Serializing
                        // the raw transcript would ship every accumulated image
                        // to the summarizer as a data URI — enormous, and the
                        // summarizer cannot use pixels in a text brief anyway.
                        let stripped: Vec<Value> = middle
                            .iter()
                            .map(|m| {
                                let mut m = m.clone();
                                if let Some(parts) = m.get("content").and_then(Value::as_array).cloned() {
                                    let text: Vec<String> = parts
                                        .iter()
                                        .map(|part| match part.get("type").and_then(Value::as_str) {
                                            Some("image_url") => "<frame omitted>".to_string(),
                                            _ => part
                                                .get("text")
                                                .and_then(Value::as_str)
                                                .unwrap_or("")
                                                .to_string(),
                                        })
                                        .collect();
                                    m["content"] = json!(text.join(" "));
                                }
                                m
                            })
                            .collect();
                        let transcript = serde_json::to_string(&stripped)
                            .unwrap_or_else(|_| "[]".into());
                        let summary_req = json!({
                            "model": model,
                            "messages": [
                                {"role": "system", "content": "You compact an agent's episode transcript. Write a dense factual brief the agent will rely on as its only memory of these turns."},
                                {"role": "user", "content": format!(
                                    "Summarize this Craftax episode segment for the agent that will keep playing. Cover: terrain and resources seen and roughly where relative to the player; inventory and achievements gained; what was attempted and failed; the current goal. Be concrete and brief. No preamble.\n\n{transcript}"
                                )}
                            ],
                            "temperature": 0.0,
                            "max_tokens": 700,
                        });
                        let summary_resp = client
                            .post(&inference_url)
                            .header("Authorization", format!("Bearer {api_key}"))
                            .header("Content-Type", "application/json")
                            .json(&summary_req)
                            .send()
                            .await;
                        if let Ok(resp) = summary_resp {
                            if let Ok(sp) = resp.json::<Value>().await {
                                accumulate_usage(&mut usage_total, sp.get("usage"));
                                let summary = extract_assistant_text(&sp);
                                if !summary.trim().is_empty() {
                                    let before = messages.len();
                                    let mut next: Vec<Value> = messages[0..head_len].to_vec();
                                    next.push(json!({
                                        "role": "user",
                                        "content": format!("[context compacted — the turns below are summarized, not verbatim]\n{summary}")
                                    }));
                                    next.extend_from_slice(&messages[start..]);
                                    messages = next;
                                    compactions.push(json!({
                                        "trigger": "fixed_threshold",
                                        "at_llm_call": llm_calls,
                                        "prompt_tokens_before": last_prompt_tokens,
                                        "messages_before": before,
                                        "messages_after": messages.len(),
                                        "summarized_messages": middle.len(),
                                        "summary_chars": summary.len(),
                                    }));
                                }
                            }
                        }
                    }
                }
            }
            policy_label = format!("craftax_react_conversation_{provider}");
        } else {
        while !engine.is_done()
            && (engine
                .private
                .get("step_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize)
                < max_steps
            && llm_calls < max_llm_turns
        {
            let steps_done = engine
                .private
                .get("step_index")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let steps_remaining = max_steps.saturating_sub(steps_done);
            let llm_remaining = max_llm_turns.saturating_sub(llm_calls);
            let batch_cap = max_actions.min(steps_remaining).max(1);
            let batch_floor = min_actions.min(batch_cap).max(1);
            let readout = engine.readout();
            let observation_text = readout
                .get("observation_text")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            // Keep the user prompt compact: full ASCII maps blow the gpt-oss
            // reasoning budget and leave `content` empty under max_tokens=512.
            let prompt = format!(
                "{objective}\n\n{observation_text}\n\nlast_actions={}\nsteps_remaining={steps_remaining}\nllm_calls_remaining={llm_remaining}\nvalid_actions={}\nPlan {batch_floor}-{batch_cap} valid actions to execute sequentially before the next observation.\nReply with JSON only, for example {{\"actions\":[\"do\",\"right\",\"do\",\"left\",\"do\"]}}.",
                serde_json::to_string(&action_history[action_history.len().saturating_sub(16)..])
                    .unwrap_or_else(|_| "[]".into()),
                valid.join(", ")
            );
            // Keep the exact decision context: a turn that reports only the
            // assistant text is not reconstructible, and an SFT example built
            // from it would train on a prompt the teacher never saw.
            let prompt_for_turn = prompt.clone();
            let observation_for_turn = observation_text.clone();
            let valid_for_turn = valid.clone();
            let mut request_body = json!({
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            });
            if provider == "deepseek" {
                request_body
                    .as_object_mut()
                    .unwrap()
                    .insert("thinking".into(), json!({"type": "disabled"}));
            }
            if let Some(effort) = reasoning_effort.as_deref() {
                request_body
                    .as_object_mut()
                    .unwrap()
                    .insert("reasoning_effort".into(), json!(effort));
            }
            if provider == "groq" && model.contains("gpt-oss") {
                let obj = request_body.as_object_mut().unwrap();
                obj.insert("reasoning_effort".into(), json!("low"));
                // Prefer leaving room for the JSON answer after CoT.
                if max_tokens < 768 {
                    obj.insert("max_tokens".into(), json!(1024));
                }
            }
            let trace_messages = trace_safe_messages(request_body
                .get("messages")
                .cloned()
                .unwrap_or_else(|| json!([])));
            let trace_steps_before = steps_done;
            let trace_reward_before = engine
                .private
                .get("total_reward")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let trace_achievements_before = engine
                .private
                .get("achievements")
                .cloned()
                .unwrap_or_else(|| json!([]));
            let payload = call_policy_lm(
                &client,
                &inference_url,
                &api_key,
                &request_body,
                policy_lm_attempts,
            ).await?;
            let assistant = extract_assistant_text(&payload);
            accumulate_usage(&mut usage_total, payload.get("usage"));
            let planned = parse_actions_batch(&assistant, &valid, batch_cap);
            let mut executed = Vec::new();
            for action in &planned {
                if engine.is_done() {
                    break;
                }
                let steps_now = engine
                    .private
                    .get("step_index")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as usize;
                if steps_now >= max_steps {
                    break;
                }
                engine
                    .step(&json!(action))
                    .map_err(|err| ApiError {
                        status: StatusCode::BAD_REQUEST,
                        code: "step_failed",
                        message: err.to_string(),
                    })?;
                executed.push(action.clone());
                action_history.push(action.clone());
            }
            if executed.is_empty() {
                // Avoid infinite loops if the model returns nothing usable.
                let fallback = valid
                    .iter()
                    .find(|a| a.as_str() == "do" || a.as_str() == "noop")
                    .cloned()
                    .unwrap_or_else(|| valid[0].clone());
                if !engine.is_done() {
                    let _ = engine.step(&json!(fallback));
                    executed.push(fallback.clone());
                    action_history.push(fallback);
                }
            }
            llm_calls += 1;
            let trace_steps_after = engine
                .private
                .get("step_index")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let trace_reward_after = engine
                .private
                .get("total_reward")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let trace_achievements_after = engine
                .private
                .get("achievements")
                .cloned()
                .unwrap_or_else(|| json!([]));
            turns.push(json!({
                "llm_call": llm_calls,
                "assistant": assistant,
                "actions": executed,
                // The user message verbatim, so a training example can reuse
                // the teacher's exact inference representation.
                "prompt": prompt_for_turn,
                "system_prompt": system_prompt,
                "observation_text": observation_for_turn,
                "valid_actions": valid_for_turn,
            }));
            trace_capture
                .emit_turn(json!({
                    "llm_call": llm_calls,
                    "messages": trace_messages,
                    "assistant_message": {"role": "assistant", "content": assistant},
                    "thinking": assistant,
                    "tool_calls": [],
                    "tool_results": [],
                    "usage": payload.get("usage").cloned().unwrap_or_else(|| json!({})),
                    "env_transition": {
                        "steps_before": trace_steps_before,
                        "steps_after": trace_steps_after,
                        "actions": executed,
                        "reward_before": trace_reward_before,
                        "reward_after": trace_reward_after,
                        "reward_delta": trace_reward_after - trace_reward_before,
                        "achievements_before": trace_achievements_before,
                        "achievements_after": trace_achievements_after,
                    },
                    "compactions": [],
                }))
                .await?;
        }
        policy_label = format!("craftax_react_{provider}");
        }
    } else {
        // Explicit smoke path when use_lm=false.
        let noop = valid
            .iter()
            .find(|name| name.eq_ignore_ascii_case("noop") || name.eq_ignore_ascii_case("do"))
            .cloned()
            .unwrap_or_else(|| valid.first().cloned().unwrap_or_else(|| "noop".into()));
        for _ in 0..max_steps {
            if engine.is_done() {
                break;
            }
            let _ = engine.step(&json!(noop));
        }
        policy_label = "craftax_heuristic_noop_smoke".into();
    }

    let reward = engine.private.get("total_reward").cloned().unwrap_or(Value::Null);
    let steps = engine.private.get("step_index").cloned().unwrap_or(Value::Null);
    let invalid_actions = engine
        .private
        .get("invalid_action_count")
        .cloned()
        .unwrap_or(Value::Null);
    let nev_ref = spool_nev(&rollout_id, &engine.events);
    let trace_ref = trace_capture.seal().await?;
    let mut record = json!({
        "status": "completed",
        "rollout_id": rollout_id,
        "outcome_reward": reward,
        "reward": reward,
        "reward_info": {"outcome_reward": reward, "score": reward},
        "summary": {
            "outcome_reward": reward,
            "env_steps": steps,
            "seed": seed,
            "policy": policy_label,
            "policy_llm_turns": llm_calls,
            "model": model,
            "provider": provider,
            "reasoning_effort": reasoning_effort,
            // The ReAct loop is stateless: each call is [system, user] and the
            // only carried state is a 16-action tail window. Nothing is
            // summarized — everything older is dropped. The Python container
            // emits `agent.context_compacted` for this; the Rust path was
            // silent, so an episode could drop 150+ actions with no record.
            "context": if react_mode == "conversation" {
                json!({
                    "mode": "conversation",
                    "strategy": "summarize_and_keep_recent",
                    "conversation_carried": true,
                    "summarized": !compactions.is_empty(),
                    "compaction_count": compactions.len(),
                    "observation_mode": observation_mode,
                    "render_tile_size": render_tile_size,
                    "keep_recent_frames": keep_recent_frames,
                    "policy_lm_attempts": policy_lm_attempts,
                    "view_radius": view_radius,
                    "scratchpad_enabled": scratchpad_enabled,
                    "scratchpad_writes": scratchpad_writes,
                    "scratchpad_chars": scratchpad.len(),
                    "scratchpad_final": scratchpad,
                    "compactions": compactions,
                    "final_prompt_tokens": last_prompt_tokens,
                    "context_token_budget": context_token_budget,
                    "compact_threshold": compact_threshold,
                    "keep_recent_messages": keep_recent_messages,
                })
            } else {
                json!({
                    "mode": "stateless",
                    "strategy": "tail_window",
                    "conversation_carried": false,
                    "summarized": false,
                    "retained_actions": action_history.len().min(16),
                    "dropped_actions": action_history.len().saturating_sub(16),
                })
            },
            "world": {
                "preset": world_preset,
                "max_steps": max_steps,
                "scaled_resources": scaled_resources,
                "is_reference_world": is_reference_world,
                "requested": world,
            },
        },
        "usage": usage_total,
        "turns": turns,
        // Paid SFT curation consumes these authoritative counters directly.
        // Missing engine evidence remains null; it is never inferred from the
        // number of turns or coerced to zero.
        "steps": steps,
        "invalid_actions": invalid_actions,
        // Spooled to disk, not inlined: a 1000-step rollout produces thousands
        // of events and nobody wants them in every response body. Retrieve via
        // GET /rollouts/{id}/events.
        //
        // Previously this shipped a literal [] while the engine — which had
        // accumulated the whole log — was dropped when the request returned.
        // The events were not merely unreported, they were destroyed, so every
        // optimizer, GEPA and SFT rollout ran with no durable step evidence.
        //
        // NOTE: this is the engine's NEV log, not Trace V5. Trace V5 emission
        // (CraftaxTrace) exists only in the Python ReAct container.
        "events": [],
        "nev": nev_ref,
        // Required by SFT curation, which fail-closes when a container does not
        // supply its own counters rather than inferring them from turn count.
        "invalid_actions": invalid_actions_total,
        "steps": steps,
        "scheduled_checkpoints": [],
        "final_achievements": engine.private.get("achievements").cloned().unwrap_or(json!([])),
        "metadata": {
            "env": "craftax-singleplayer",
            "seed": seed,
            "task": "craftax",
            "backend": "gamebench_gold_rust",
            "optimizer_compatible": true
        },
        "terminated": true,
        "truncated": engine.private.get("truncated").and_then(Value::as_bool).unwrap_or(false),
    });
    // Trace V5 is independently stored by synth-containers. Only the digest/path
    // manifest crosses the rollout response boundary. Capture-off omits the field
    // entirely so the default response contract remains byte-for-byte shaped as it
    // was before tracing support.
    if trace_ref.get("capture").and_then(Value::as_str) != Some("off") {
        record["summary"]["trace"] = trace_ref.clone();
    }
    let trace_v5_digest = trace_ref.get("trace_v5_digest").cloned().unwrap_or(Value::Null);
    record["sealed"] = json!(trace_v5_digest.is_string());
    record["trace_v5_digest"] = trace_v5_digest;
    let session = RolloutSession {
        _admission_permit: admission_permit,
        engine,
        started_at: Instant::now(),
        checkpoints: HashMap::new(),
        frames: HashMap::new(),
        gif_frames: HashMap::new(),
        terminal_record: Some(record.clone()),
        status: "completed".into(),
    };
    app.sessions.lock().unwrap().insert(rollout_id.clone(), Arc::new(Mutex::new(session)));
    Ok(Json(record))
}

fn default_inference_url(provider: &str) -> String {
    match provider {
        "openrouter" => "https://openrouter.ai/api/v1/chat/completions".into(),
        "deepseek" => "https://api.deepseek.com/chat/completions".into(),
        "openai" => "https://api.openai.com/v1/chat/completions".into(),
        "gemini" => {
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions".into()
        }
        _ => "https://api.groq.com/openai/v1/chat/completions".into(),
    }
}

fn default_api_key_env(provider: &str) -> String {
    match provider {
        "openrouter" => "OPENROUTER_API_KEY".into(),
        "deepseek" => "DEEPSEEK_API_KEY".into(),
        "openai" => "OPENAI_API_KEY".into(),
        "gemini" => "GEMINI_API_KEY".into(),
        _ => "GROQ_API_KEY".into(),
    }
}

fn extract_assistant_text(payload: &Value) -> String {
    let message = payload.pointer("/choices/0/message");
    let content = message
        .and_then(|m| m.get("content"))
        .cloned()
        .unwrap_or(Value::Null);
    let mut text = match content {
        Value::String(s) => s,
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    if text.trim().is_empty() {
        // gpt-oss on Groq often puts the entire reply in `reasoning` when the
        // completion budget is consumed by chain-of-thought.
        if let Some(reasoning) = message
            .and_then(|m| m.get("reasoning").or_else(|| m.get("reasoning_content")))
            .and_then(Value::as_str)
        {
            text = reasoning.to_string();
        }
    }
    if let Some(json_blob) = extract_embedded_json_object(&text) {
        return json_blob;
    }
    text
}

fn extract_embedded_json_object(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.starts_with('{') && serde_json::from_str::<Value>(trimmed).is_ok() {
        return Some(trimmed.to_string());
    }
    // Prefer fenced ```json ... ``` blocks, then the last {...} slice.
    if let Some(start) = trimmed.find("```json") {
        let after = &trimmed[start + 7..];
        if let Some(end) = after.find("```") {
            let inner = after[..end].trim();
            if serde_json::from_str::<Value>(inner).is_ok() {
                return Some(inner.to_string());
            }
        }
    }
    let mut last = None;
    let bytes = trimmed.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] != b'{' {
            continue;
        }
        let mut depth = 0i32;
        for j in i..bytes.len() {
            match bytes[j] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        let slice = &trimmed[i..=j];
                        if let Ok(v) = serde_json::from_str::<Value>(slice) {
                            if v.get("actions").is_some() || v.get("action").is_some() {
                                last = Some(slice.to_string());
                            }
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    last
}

fn accumulate_usage(total: &mut Value, usage: Option<&Value>) {
    let Some(usage) = usage.and_then(Value::as_object) else {
        return;
    };
    let Some(obj) = total.as_object_mut() else {
        return;
    };
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        let add = usage.get(key).and_then(Value::as_u64).unwrap_or(0);
        let cur = obj.get(key).and_then(Value::as_u64).unwrap_or(0);
        obj.insert(key.into(), json!(cur + add));
    }
}

/// Parse a batch and report how many emitted actions were not valid action
/// names. The count is required evidence downstream: SFT curation hard-rejects
/// a trajectory whose container did not supply its invalid-action counter,
/// because a policy that mostly emits garbage is not a demonstration.
fn parse_actions_batch_counted(
    raw: &str,
    valid: &[String],
    batch_cap: usize,
) -> (Vec<String>, usize) {
    let emitted = count_emitted_actions(raw);
    let parsed = parse_actions_batch(raw, valid, batch_cap);
    // Only actions beyond the cap are legitimately dropped; the rest were
    // unrecognized names.
    let kept_or_capped = parsed.len().max(emitted.min(batch_cap));
    (parsed, emitted.saturating_sub(kept_or_capped))
}

/// How many action strings the model actually emitted, valid or not.
fn count_emitted_actions(raw: &str) -> usize {
    let text = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
        if let Some(arr) = parsed.get("actions").and_then(Value::as_array) {
            return arr.iter().filter(|v| v.is_string()).count();
        }
        if let Some(arr) = parsed.as_array() {
            return arr.iter().filter(|v| v.is_string()).count();
        }
        if parsed.get("action").or_else(|| parsed.get("move")).or_else(|| parsed.get("command")).is_some() {
            return 1;
        }
    }
    0
}

fn parse_actions_batch(raw: &str, valid: &[String], batch_cap: usize) -> Vec<String> {
    let text = raw.trim();
    let mut out = Vec::new();
    if let Ok(parsed) = serde_json::from_str::<Value>(text) {
        if let Some(arr) = parsed.get("actions").and_then(Value::as_array) {
            for item in arr {
                if let Some(action) = resolve_action(item.as_str().unwrap_or(""), valid) {
                    out.push(action);
                }
            }
        } else if let Some(arr) = parsed.as_array() {
            for item in arr {
                if let Some(action) = resolve_action(item.as_str().unwrap_or(""), valid) {
                    out.push(action);
                }
            }
        } else if let Some(single) = parsed
            .get("action")
            .or_else(|| parsed.get("move"))
            .or_else(|| parsed.get("command"))
            .and_then(Value::as_str)
        {
            if let Some(action) = resolve_action(single, valid) {
                out.push(action);
            }
        }
    }
    if out.is_empty() {
        // Last resort: prefer a productive default over scanning free-form CoT,
        // which casually mentions many valid action names.
        if let Some(action) = resolve_action("do", valid).or_else(|| valid.first().cloned()) {
            out.push(action);
        }
    }
    if out.len() > batch_cap {
        out.truncate(batch_cap);
    }
    out
}

fn resolve_action(raw: &str, valid: &[String]) -> Option<String> {
    let cleaned = raw
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c == ',' || c == ';')
        .to_ascii_lowercase();
    if cleaned.is_empty() {
        return None;
    }
    valid
        .iter()
        .find(|a| a.eq_ignore_ascii_case(&cleaned))
        .cloned()
}

async fn get_rollout_record(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    if let Some(record) = &session.terminal_record {
        return Ok(Json(record.clone()));
    }
    let mut payload = rollout_payload(&rollout_id, session);
    payload["status"] = json!(&session.status);
    Ok(Json(payload))
}

async fn rollout_state(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "status": session.status,
        "terminated": session.status == "completed" || session.status == "failed" || session.status == "cancelled",
    })))
}

async fn terminate_rollout(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let mut guard = handle.lock().unwrap();
    let session = &mut *guard;
    session.status = "cancelled".into();
    Ok(Json(json!({"rollout_id": rollout_id, "status": "cancelled"})))
}

/// GELO resume: parent may be missing after heuristic sync completion; fall back
/// to a fresh optimizer episode using request overrides (smoke-compatible).
async fn resume_optimizer_rollout(
    State(app): State<AppState>,
    Path(_parent_rollout_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let mut overrides = body
        .get("overrides")
        .cloned()
        .unwrap_or_else(|| body.clone());
    if let Some(object) = overrides.as_object_mut() {
        if let Some(target) = body
            .get("target_rollout_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .filter(|value| !value.is_empty())
        {
            object.insert("rollout_id".into(), json!(target));
        }
        object
            .entry("submission_mode".to_string())
            .or_insert(json!("sync"));
    }
    run_optimizer_rollout(app, overrides).await
}

async fn gepa_dataset() -> Json<Value> {
    Json(json!({
        "schema": "synth.container.dataset.v1",
        "task_family": "craftax-singleplayer",
        "splits": {
            "train": {"count": 8},
            "validation": {"count": 4},
            "test": {"count": 4}
        },
        "rows": (101_i64..=108).map(|seed| json!({
            "example_id": format!("craftax:train:{seed}"),
            "seed": seed,
            "split": "train",
            "input": {"seed": seed, "task": "craftax"},
            "label": null
        })).collect::<Vec<_>>()
    }))
}

async fn gepa_dataset_rows(Json(body): Json<Value>) -> Json<Value> {
    let split = body
        .get("split")
        .and_then(Value::as_str)
        .unwrap_or("train");
    let seeds: Vec<i64> = body
        .get("seeds")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_i64)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| match split {
            "validation" => (201..=204).collect(),
            "test" => (301..=304).collect(),
            _ => (101..=108).collect(),
        });
    Json(json!({
        "schema": "synth.container.dataset_rows.v1",
        "split": split,
        "rows": seeds.into_iter().map(|seed| json!({
            "example_id": format!("craftax:{split}:{seed}"),
            "task_id": format!("craftax:{split}:{seed}"),
            "seed": seed,
            "split": split,
            "input": {"seed": seed, "task": "craftax"},
            "label": null
        })).collect::<Vec<_>>()
    }))
}

async fn gepa_program() -> Json<Value> {
    let seed_prompt =
        "You are playing Craftax. Prefer survival and crafting progress.";
    Json(json!({
        "schema_version": "prompt_program.v1",
        "modules": [{
            "module_id": "react_system_prompt",
            "role": "system",
            "content": seed_prompt,
            "mutable": true,
            "candidate_field": "react_system_prompt"
        }],
        "target_modules": [{
            "module_id": "react_system_prompt",
            "candidate_field": "react_system_prompt",
            "objective": "outcome_reward"
        }],
        "seed_candidate": {
            "react_system_prompt": seed_prompt
        },
        "mutable_keys": ["react_system_prompt"],
        "metadata": {
            "task_family": "craftax-singleplayer",
            "task": "craftax"
        }
    }))
}

async fn optimizer_metadata() -> Json<Value> {
    Json(json!({
        "env_family": "craftax-singleplayer",
        "optimizer_targets": ["go_ex", "gepa"],
        "gelo_tier": "A",
        "gepa_tier": "smoke",
        "optimizer_contracts": {
            "gepa": "synth_optimizers.gepa.v2",
            "go_ex": "synth_optimizers.go_ex.v1"
        }
    }))
}

async fn compatibility(Query(query): Query<HashMap<String, String>>) -> Json<Value> {
    let target = query.get("target").map(String::as_str).unwrap_or("");
    Json(json!({
        "target": target,
        "compatible": matches!(target, "go_ex" | "go-ex" | "gelo" | "gepa" | ""),
        "env_family": "craftax-singleplayer"
    }))
}

async fn step(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<StepRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let mut guard = handle.lock().unwrap();
    let session = &mut *guard;
    // The engine already knows why an action did what it did — it labels every
    // step `move`, `blocked`, `harvest`, `place`, or `noop` with a reason. Those
    // records stayed in the event log, so callers had to infer effect by
    // diffing observations, and reported roughly 45 of 100 actions as
    // "no observable core-state effect" with no idea which were refusals,
    // which were blocked, and which genuinely did nothing.
    let before_cursor = session.engine.events.len();
    if !session.engine.is_done() {
        session.engine.step(&body.action).map_err(bad_request)?;
        let telemetry_enabled = app.telemetry.lock().unwrap().contains_key(&rollout_id);
        capture_frame_if_enabled(session, app.replay, telemetry_enabled);
    }
    let mut payload = rollout_payload(&rollout_id, session);
    payload["outcome"] = step_outcome(&body.action, &session.engine.events[before_cursor..]);
    if let Some(sender) = app.telemetry.lock().unwrap().get(&rollout_id) {
        let event = rollout_event(&rollout_id, &payload);
        let mut histories = app.telemetry_history.lock().unwrap();
        let history = histories.entry(rollout_id.clone()).or_default();
        history.push(event.clone());
        if history.len() > 2048 { history.remove(0); }
        let _ = sender.send(event);
    }
    Ok(Json(payload))
}

/// What one action actually did, in the engine's own words.
///
/// `effect` is the engine's transition label for the action itself: `move`,
/// `blocked`, `blocked_by_mob`, `harvest`, `place`, `noop`, and so on. `reason`
/// is the engine's own explanation when it refused — `needs_pickaxe:stone`,
/// `missing_resources`, `needs_crafting_table`, `nothing_to_do:grass`. A caller
/// no longer has to guess from an observation diff whether an action was
/// refused, obstructed, or simply pointless.
///
/// `changed` is the honest summary: false means the engine itself recorded no
/// state change, not that an observer failed to notice one.
fn step_outcome(action: &Value, events: &[EventRecord]) -> Value {
    let applied = events
        .iter()
        .find(|event| event.kind == "action_applied")
        .or_else(|| events.iter().find(|event| event.kind == "floor_transition"));
    let effect = applied
        .and_then(|event| event.transition.clone())
        .unwrap_or_else(|| "unrecorded".to_string());
    let reason = applied
        .and_then(|event| event.payload.get("reason"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let resource_deltas: Vec<Value> = events
        .iter()
        .filter(|event| event.kind == "resource_delta")
        .map(|event| {
            json!({
                "resource": event.transition,
                "before": event.payload.get("before"),
                "after": event.payload.get("after"),
                "delta": event.payload.get("delta"),
            })
        })
        .collect();
    let achievements: Vec<Value> = events
        .iter()
        .filter(|event| event.kind == "achievement_unlocked")
        .filter_map(|event| event.payload.get("achievement").cloned())
        .collect();
    let violations: Vec<Value> = events
        .iter()
        .filter(|event| event.kind == "rule_violation")
        .map(|event| event.payload.clone())
        .collect();
    // `changed` follows the evidence, not the vocabulary. A transition the
    // engine calls `blocked` moved nothing, and a label this function has never
    // seen is reported as unknown rather than guessed either way — guessing is
    // how "no observable core-state effect" became a category that swallowed
    // refusals, obstructions, and genuine no-ops alike.
    const NO_EFFECT: [&str; 4] = ["noop", "blocked", "blocked_by_mob", "boss_not_vulnerable"];
    const EFFECT: [&str; 11] = [
        "move",
        "harvest",
        "collect_sapling",
        "drink",
        "eat_plant",
        "open_chest",
        "mine_block",
        "place",
        "recover",
        "boss_damage",
        "intrinsic_mode_request",
    ];
    let changed = if !resource_deltas.is_empty() || !achievements.is_empty() {
        json!(true)
    } else if NO_EFFECT.contains(&effect.as_str()) {
        json!(false)
    } else if EFFECT.contains(&effect.as_str()) {
        json!(true)
    } else if effect == "unrecorded" {
        json!(false)
    } else {
        Value::Null
    };
    json!({
        "action": action.clone(),
        "effect": effect,
        "reason": reason,
        "changed": changed,
        "detail": applied.map(|event| event.payload.clone()),
        "resource_deltas": resource_deltas,
        "achievements_unlocked": achievements,
        "rule_violations": violations,
        "event_count": events.len(),
    })
}

fn rollout_event(rollout_id: &str, payload: &Value) -> Value {
    let mut streamed_payload = payload.clone();
    let step = payload.pointer("/progress/env_steps").and_then(Value::as_u64).unwrap_or(0);
    streamed_payload["frame_url"] = json!(format!("/rollouts/{rollout_id}/frames/{step}.png"));
    json!({
        "schema": "synth.rollout.event.v1",
        "rollout_id": rollout_id,
        "run_id": rollout_id,
        "lane": rollout_id,
        "kind": if payload.get("terminated").and_then(Value::as_bool).unwrap_or(false) || payload.get("truncated").and_then(Value::as_bool).unwrap_or(false) { "eval.run.terminal" } else { "snapshot" },
        "event_id": payload.pointer("/progress/env_steps").cloned().unwrap_or(json!(0)),
        "ts": chrono::Utc::now().to_rfc3339(),
        "payload": streamed_payload
    })
}

async fn rollout_stream(State(app): State<AppState>, Path(rollout_id): Path<String>) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let receiver = app.telemetry.lock().unwrap().get(&rollout_id).map(broadcast::Sender::subscribe).ok_or_else(|| ApiError { status: StatusCode::NOT_FOUND, code: "telemetry_not_enabled", message: format!("telemetry not enabled for rollout: {rollout_id}") })?;
    let history = app.telemetry_history.lock().unwrap().get(&rollout_id).cloned().unwrap_or_default();
    let historical = tokio_stream::iter(history).map(|event| Ok(sse_event(event)));
    let live = BroadcastStream::new(receiver).filter_map(|message| message.ok().map(|event| Ok(sse_event(event))));
    Ok(Sse::new(historical.chain(live)).keep_alive(KeepAlive::default()))
}

fn sse_event(event: Value) -> Event {
        let event_id = event["event_id"].to_string();
        Event::default().id(event_id).event(event["kind"].as_str().unwrap_or("snapshot")).json_data(event).expect("event JSON")
}

async fn rollout_ws(ws: WebSocketUpgrade, State(app): State<AppState>, Path(rollout_id): Path<String>) -> Result<Response, ApiError> {
    let receiver = app.telemetry.lock().unwrap().get(&rollout_id).map(broadcast::Sender::subscribe).ok_or_else(|| ApiError { status: StatusCode::NOT_FOUND, code: "telemetry_not_enabled", message: format!("telemetry not enabled for rollout: {rollout_id}") })?;
    let history = app.telemetry_history.lock().unwrap().get(&rollout_id).cloned().unwrap_or_default();
    Ok(ws.on_upgrade(move |socket| stream_websocket(socket, receiver, history)).into_response())
}

async fn stream_websocket(mut socket: WebSocket, mut receiver: broadcast::Receiver<Value>, history: Vec<Value>) {
    for event in history {
        if socket.send(Message::Text(event.to_string())).await.is_err() { return; }
    }
    while let Ok(event) = receiver.recv().await {
        if socket.send(Message::Text(event.to_string())).await.is_err() { break; }
    }
}

async fn delete_rollout(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let mut guard = app.sessions.lock().unwrap();
    if guard.remove(&rollout_id).is_none() {
        return Err(rollout_not_found());
    }
    drop(guard);
    app.telemetry.lock().unwrap().remove(&rollout_id);
    app.telemetry_history.lock().unwrap().remove(&rollout_id);
    Ok(Json(json!({"rollout_id": rollout_id, "deleted": true})))
}

async fn readout(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    Ok(Json(session.engine.readout()))
}

async fn event_log(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    // A live session wins; otherwise fall back to the spool, which is the only
    // record an optimizer rollout leaves behind.
    let live = {
        session_handle(&app, &rollout_id).map(|handle| {
            let session = &*handle.lock().unwrap();
            json!({
                "rollout_id": rollout_id,
                "source": "session",
                "events": session.engine.events,
                "legacy": session.engine.legacy_strings(),
                "nev_cursor": session.engine.events.len(),
            })
        })
    };
    if let Some(payload) = live {
        return Ok(Json(payload));
    }
    let path = nev_dir().join(format!("{rollout_id}.jsonl"));
    if let Ok(body) = std::fs::read_to_string(&path) {
        let events: Vec<Value> = body
            .lines()
            .filter(|line| !line.trim().is_empty())
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect();
        return Ok(Json(json!({
            "rollout_id": rollout_id,
            "source": "spool",
            "path": path.to_string_lossy(),
            "nev_cursor": events.len(),
            "events": events,
        })));
    }
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "events": session.engine.events,
        "legacy": session.engine.legacy_strings(),
        "nev_cursor": session.engine.events.len()
    })))
}

async fn checkpoint(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let mut guard = handle.lock().unwrap();
    let session = &mut *guard;
    let blob = session.engine.checkpoint_bytes().map_err(bad_request)?;
    let checkpoint_id = Uuid::new_v4().to_string();
    session
        .checkpoints
        .insert(checkpoint_id.clone(), blob.clone());
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "checkpoint_id": checkpoint_id,
        "blob": BASE64.encode(&blob),
        "bytes": blob.len(),
        "step_index": session.engine.private.get("step_index").cloned().unwrap_or(Value::Null),
        "nev_cursor": session.engine.events.len(),
        "config_hash": session.engine.resolved.config_hash
    })))
}

async fn restore(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<RestoreRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let mut guard = handle.lock().unwrap();
    let session = &mut *guard;
    let blob = BASE64.decode(body.blob).map_err(bad_request)?;
    let restored = session
        .engine
        .restore_checkpoint_bytes(&blob)
        .map_err(bad_request)?;
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "restore_report": {
            "bytes": blob.len(),
            "wall_ms": 0.0,
            "nev_events_restored": restored
        },
        "state": rollout_payload(&rollout_id, session)
    })))
}

async fn simulate(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
    Json(body): Json<SimulateRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    let root = BASE64.decode(body.blob).map_err(bad_request)?;
    let mut results = Vec::new();
    for (index, sequence) in body.sequences.iter().enumerate() {
        let mut sim = session.engine.clone();
        sim.restore_checkpoint_bytes(&root).map_err(bad_request)?;
        let root_invalid_action_count = sim
            .private
            .get("invalid_action_count")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let mut reward_trace = Vec::new();
        for action in sequence {
            if sim.is_done() {
                break;
            }
            sim.step(&Value::String(action.clone()))
                .map_err(bad_request)?;
            reward_trace.push(
                sim.private
                    .get("reward_last")
                    .cloned()
                    .unwrap_or(json!(0.0)),
            );
        }
        results.push(json!({
            "index": index,
            "actions": sequence,
            "reward": sim.private.get("total_reward").cloned().unwrap_or(json!(0.0)),
            "reward_trace": reward_trace,
            "invalid_action_delta": sim
                .private
                .get("invalid_action_count")
                .and_then(Value::as_i64)
                .unwrap_or(0)
                - root_invalid_action_count,
            "achievements": sim.private.get("achievements").cloned().unwrap_or_else(|| json!([])),
            "terminated": sim.private.get("terminated").and_then(Value::as_bool).unwrap_or(false),
            "truncated": sim.private.get("truncated").and_then(Value::as_bool).unwrap_or(false),
            "steps": sim.private.get("step_index").cloned().unwrap_or(json!(0)),
            "readout": sim.readout()
        }));
    }
    Ok(Json(json!({
        "rollout_id": rollout_id,
        "root_nev_cursor": session.engine.events.len(),
        "results": results
    })))
}

async fn render_svg(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Response, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    let ascii = session
        .engine
        .readout()
        .get("ascii")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    Ok((
        [(header::CONTENT_TYPE, "image/svg+xml")],
        svg_for_ascii(&ascii),
    )
        .into_response())
}

async fn render_png_route(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Response, ApiError> {
    let png = {
        let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
        let guard = handle.lock().unwrap();
        let session = &*guard;
        let frame = render_rgb_frame_from_world(
            &session.engine.world,
            DEFAULT_RENDER_TILE_SIZE,
            RenderMode::Auto,
        );
        encode_png_rgb(frame.0, frame.1, &frame.2)
    };
    Ok(([(header::CONTENT_TYPE, "image/png")], png).into_response())
}

async fn observation_png_route(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Response, ApiError> {
    let png = {
        let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
        let guard = handle.lock().unwrap();
        let session = &*guard;
        let (w, h, rows) = craftax_gamebench_gold::render_observation_frame(
            &session.engine.world,
            4,
            DEFAULT_RENDER_TILE_SIZE * 2,
        );
        encode_png_rgb(w, h, &rows)
    };
    Ok(([(header::CONTENT_TYPE, "image/png")], png).into_response())
}

async fn frame_manifest_route(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
    let guard = handle.lock().unwrap();
    let session = &*guard;
    Ok(Json(frame_manifest(&rollout_id, session)))
}

async fn frame_png_route(
    State(app): State<AppState>,
    Path((rollout_id, step_raw)): Path<(String, String)>,
) -> Result<Response, ApiError> {
    let step = parse_frame_step(&step_raw).ok_or_else(|| ApiError {
        status: StatusCode::BAD_REQUEST,
        code: "frame_step_invalid",
        message: "frame step must be a non-negative integer".into(),
    })?;
    let png = {
        let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
        let mut guard = handle.lock().unwrap();
        let session = &mut *guard;
        let current_step = session_step_index(session);
        if step == current_step && !session.frames.contains_key(&step) {
            store_frame(session, step);
        }
        session.frames.get(&step).cloned()
    };
    let Some(png) = png else {
        return Err(ApiError {
            status: StatusCode::NOT_FOUND,
            code: "frame_not_found",
            message: "frame not found".into(),
        });
    };
    Ok(([(header::CONTENT_TYPE, "image/png")], png).into_response())
}

async fn replay_gif_route(
    State(app): State<AppState>,
    Path(rollout_id): Path<String>,
    Query(query): Query<ReplayGifQuery>,
) -> Result<Response, ApiError> {
    let gif = {
        let handle = session_handle(&app, &rollout_id).ok_or_else(rollout_not_found)?;
        let mut guard = handle.lock().unwrap();
        let session = &mut *guard;
        if session.gif_frames.is_empty() {
            store_frame(session, session_step_index(session));
        }
        let mut steps: Vec<u64> = session
            .gif_frames
            .keys()
            .copied()
            .filter(|step| query.through_step.is_none_or(|limit| *step <= limit))
            .collect();
        steps.sort_unstable();
        if steps.is_empty() {
            store_frame(session, session_step_index(session));
            steps.push(session_step_index(session));
        }
        let frames: Vec<RgbFrame> = steps
            .iter()
            .filter_map(|step| session.gif_frames.get(step).cloned())
            .collect();
        encode_gif_via_ffmpeg(&frames, 10).map_err(|err| ApiError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            code: "gif_encode_failed",
            message: err,
        })?
    };
    Ok(([(header::CONTENT_TYPE, "image/gif")], gif).into_response())
}

fn replay_enabled_from_env() -> bool {
    matches!(
        std::env::var("GAMEBENCH_CRAFTAX_REPLAY_ENABLED")
            .unwrap_or_default()
            .as_str(),
        "1" | "true" | "TRUE" | "yes" | "YES"
    )
}

fn parse_frame_step(raw: &str) -> Option<u64> {
    raw.strip_suffix(".png")
        .or(Some(raw))
        .and_then(|value| value.parse().ok())
}

/// Clone out the handle for `rollout_id`, holding the map lock only for the
/// lookup itself.
///
/// The map lock used to be held for whole handlers -- engine step, frame
/// render, payload build, telemetry fan-out -- so every concurrent stepper
/// queued behind every other one. Measured against the real engine, per-step
/// latency rose linearly with concurrency (1.5ms at 1, 25ms at 32) while
/// throughput stayed pinned near 1,250 steps/s no matter how many callers
/// arrived.
fn session_handle(app: &AppState, rollout_id: &str) -> Option<SessionHandle> {
    app.sessions.lock().unwrap().get(rollout_id).cloned()
}

fn session_step_index(session: &RolloutSession) -> u64 {
    session
        .engine
        .private
        .get("step_index")
        .and_then(Value::as_u64)
        .unwrap_or(0)
}

fn rollout_capture_enabled(session: &RolloutSession) -> bool {
    let readouts = &session.engine.resolved.readouts;
    let visual = readouts
        .get("visual")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let stream = readouts.get("stream").and_then(Value::as_object);
    let stream_enabled = stream
        .and_then(|stream| stream.get("enabled"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let persist_frames = stream
        .and_then(|stream| stream.get("persist_frames"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    visual || stream_enabled || persist_frames
}

fn capture_frame_if_enabled(session: &mut RolloutSession, replay: ReplaySettings, force: bool) {
    if !replay.enabled {
        return;
    }
    if !force && !rollout_capture_enabled(session) {
        return;
    }
    store_frame(session, session_step_index(session));
}

fn store_frame(session: &mut RolloutSession, step: u64) {
    let frame = render_rgb_frame_from_world(
        &session.engine.world,
        DEFAULT_RENDER_TILE_SIZE,
        RenderMode::Auto,
    );
    let png = encode_png_rgb(frame.0, frame.1, &frame.2);
    session.frames.insert(step, png);
    session.gif_frames.insert(step, frame);
}

fn frame_manifest(rollout_id: &str, session: &RolloutSession) -> Value {
    let mut frames = Vec::new();
    for step in session.frames.keys().copied().collect::<Vec<_>>() {
        let blob = session.frames.get(&step).expect("frame step");
        frames.push(json!({
            "step": step,
            "bytes": blob.len(),
            "sha256": frame_sha256(blob),
            "url": format!("/rollouts/{rollout_id}/frames/{step}.png"),
        }));
    }
    frames.sort_by_key(|entry| entry.get("step").and_then(Value::as_u64).unwrap_or(0));
    json!({
        "rollout_id": rollout_id,
        "frame_count": frames.len(),
        "latest_step": frames.last().and_then(|entry| entry.get("step")).cloned(),
        "frames": frames,
    })
}

fn rollout_payload(rollout_id: &str, session: &RolloutSession) -> Value {
    json!({
        "rollout_id": rollout_id,
        "readout": session.engine.readout(),
        "reward": session.engine.private.get("total_reward").cloned().unwrap_or(json!(0.0)),
        "terminated": session.engine.private.get("terminated").and_then(Value::as_bool).unwrap_or(false),
        "truncated": session.engine.private.get("truncated").and_then(Value::as_bool).unwrap_or(false),
        "nev_cursor": session.engine.events.len(),
        "progress": {
            "env_steps": session.engine.private.get("step_index").cloned().unwrap_or(json!(0)),
            "wall_clock_seconds": session.started_at.elapsed().as_secs_f64()
        }
    })
}

fn default_task() -> Value {
    serde_json::from_str(DEFAULT_TASK).expect("default task JSON is valid")
}

fn bad_request(error: impl std::fmt::Display) -> ApiError {
    ApiError {
        status: StatusCode::BAD_REQUEST,
        code: "bad_request",
        message: error.to_string(),
    }
}

fn rollout_not_found() -> ApiError {
    ApiError {
        status: StatusCode::NOT_FOUND,
        code: "rollout_not_found",
        message: "rollout not found".into(),
    }
}

fn svg_for_ascii(ascii: &str) -> String {
    let escaped = ascii
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;");
    let lines = ascii.lines().count().max(1);
    let cols = ascii.lines().map(str::len).max().unwrap_or(1).max(1);
    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}"><rect width="100%" height="100%" fill="#111"/><text x="12" y="20" fill="#e8e8e8" font-family="monospace" font-size="14"><tspan x="12" dy="0">{}</tspan></text></svg>"##,
        cols * 9 + 24,
        lines * 18 + 24,
        escaped.replace('\n', r#"</tspan><tspan x="12" dy="18">"#)
    )
}

#[cfg(test)]
mod outcome_tests {
    use super::*;

    fn event(kind: &str, transition: &str, payload: Value) -> EventRecord {
        EventRecord {
            step_index: 1,
            tick: 1,
            episode_id: "episode".into(),
            kind: kind.into(),
            action: Some(json!("do")),
            transition: Some(transition.into()),
            severity: "info".into(),
            message: String::new(),
            payload,
        }
    }

    /// The engine knows a refused craft from a blocked move. Callers were left
    /// diffing observations and calling both "no observable effect".
    #[test]
    fn a_refusal_carries_the_engines_own_reason() {
        let outcome = step_outcome(
            &json!("make_stone_pickaxe"),
            &[event(
                "action_applied",
                "noop",
                json!({"reason": "missing_resources", "costs": {"stone": 1}}),
            )],
        );
        assert_eq!(outcome["effect"], "noop");
        assert_eq!(outcome["reason"], "missing_resources");
        assert_eq!(outcome["changed"], false);
        assert_eq!(outcome["detail"]["costs"]["stone"], 1);
    }

    #[test]
    fn a_blocked_move_is_not_the_same_as_a_pointless_one() {
        let blocked = step_outcome(
            &json!("left"),
            &[event(
                "action_applied",
                "blocked",
                json!({"reason": "blocked:stone", "target": [3, 4]}),
            )],
        );
        assert_eq!(blocked["effect"], "blocked");
        assert_eq!(blocked["reason"], "blocked:stone");
        assert_eq!(blocked["changed"], false);

        let moved = step_outcome(
            &json!("left"),
            &[event("action_applied", "move", json!({"to": [3, 4]}))],
        );
        assert_eq!(moved["effect"], "move");
        assert_eq!(moved["changed"], true);
        assert!(moved["reason"].is_null());
    }

    #[test]
    fn a_harvest_reports_the_resource_it_moved() {
        let outcome = step_outcome(
            &json!("do"),
            &[
                event("action_applied", "harvest", json!({"resource": "wood"})),
                event(
                    "resource_delta",
                    "wood",
                    json!({"resource": "wood", "before": 0, "after": 1, "delta": 1}),
                ),
                event(
                    "achievement_unlocked",
                    "collect_wood",
                    json!({"achievement": "collect_wood"}),
                ),
            ],
        );
        assert_eq!(outcome["effect"], "harvest");
        assert_eq!(outcome["changed"], true);
        assert_eq!(outcome["resource_deltas"][0]["resource"], "wood");
        assert_eq!(outcome["resource_deltas"][0]["delta"], 1);
        assert_eq!(outcome["achievements_unlocked"][0], "collect_wood");
    }

    /// A step the engine recorded nothing for says so, rather than borrowing
    /// the label of an action that did something.
    #[test]
    fn an_unrecorded_step_is_named_unrecorded() {
        let outcome = step_outcome(&json!("noop"), &[]);
        assert_eq!(outcome["effect"], "unrecorded");
        assert_eq!(outcome["changed"], false);
        assert_eq!(outcome["event_count"], 0);
    }
}

#[cfg(test)]
mod catalog_tests {
	use super::*;

	#[test]
	fn fixed_seed_catalog_has_stable_train_and_test_instances() {
		let catalog = task_catalog_value();
		let instances = catalog["instances"].as_array().expect("instances");
		assert_eq!(instances.len(), 40);
		assert_eq!(instances[0]["task_instance_id"], "craftax:train:1001");
		assert_eq!(instances[31]["task_instance_id"], "craftax:train:1032");
		assert_eq!(instances[32]["task_instance_id"], "craftax:test:2001");
		assert_eq!(instances[39]["metadata"]["seed"], 2008);
	}

	#[tokio::test]
	async fn capture_defaults_off_without_a_control_service() {
		let capture = TraceCapture::open(
			&json!({}),
			"rollout-off",
			reqwest::Client::new(),
		)
		.await
		.expect("capture off");
		assert_eq!(capture.requested, "off");
		assert!(capture.capture_id.is_none());
	}

	#[tokio::test]
	async fn required_capture_rejects_unavailable_control_service() {
		let result = TraceCapture::open(
			&json!({"capture": "required", "capture_url": "http://127.0.0.1:0"}),
			"rollout-required",
			reqwest::Client::new(),
		)
		.await;
		assert!(result.is_err());
		let error = result.err().expect("required error");
		assert_eq!(error.code, "trace_capture_required");
	}

	#[test]
	fn trace_omits_opaque_provider_reasoning_continuations() {
		let safe = trace_safe_messages(json!([{
			"role": "assistant",
			"content": "exposed reasoning",
			"reasoning_details": [{"data": "opaque-sk-not-a-secret"}],
		}]));
		assert_eq!(safe[0]["content"], "exposed reasoning");
		assert!(safe[0].get("reasoning_details").is_none());
	}
}

#[cfg(test)]
mod admission_tests {
    use super::*;

    fn app_with_capacity(max_sessions: usize) -> AppState {
        AppState {
            sessions: Arc::new(Mutex::new(HashMap::new())),
            telemetry: Arc::new(Mutex::new(HashMap::new())),
            telemetry_history: Arc::new(Mutex::new(HashMap::new())),
            session_slots: Arc::new(Semaphore::new(max_sessions)),
            max_sessions,
            replay: ReplaySettings { enabled: false },
        }
    }

    #[test]
    fn session_capacity_rejects_then_returns_on_drop() {
        let app = app_with_capacity(1);
        let permit = reserve_session_slot(&app).expect("first slot");
        let error = reserve_session_slot(&app).expect_err("capacity must reject");
        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.code, "too_many_sessions");
        drop(permit);
        assert!(reserve_session_slot(&app).is_ok());
    }
}
