//! HTTP game transport (`POST /api/game/v1/turn`) — the streaming successor to the
//! file-based NVBridge folder, all in Rust.
//!
//! Today the game talks to the backend by writing request files the bridge polls,
//! and reading response/audio files back (the `fnv_bridge` in-process fold runs that
//! loop inside chasm). This endpoint is the same NPC turn over HTTP: the body builds
//! a [`NativeRequest`], and [`run_turn_with_sink`](chasm_fnv_bridge::run_turn_with_sink)
//! drives the existing orchestration with an [`HttpStreamSink`] that streams the
//! turn's outputs back as NDJSON instead of writing files.
//!
//! A C++ plugin (in a SEPARATE repo) will later become the HTTP client; this is only
//! the Rust backend half. The file bridge stays the default and is untouched — this
//! is an additive, separate `/api/game/*` namespace.
//!
//! Event shapes mirror the names the bridge's own chasm client consumes
//! (`speech.delta`, `audio.chunk` with `audio.data`/`mimeType`/`captionMaxChars`),
//! plus a structured `reply`, an `action`, and a terminal `turn.completed`.

use std::sync::{Arc, Mutex};

use axum::{body::Body, extract::State, http::header, response::IntoResponse, Json};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use chasm_core::AppSettings;
use chasm_fnv_bridge::actions::{ActionActor, GameMaster};
use chasm_fnv_bridge::chasm::{AudioChunk, ChasmClient};
use chasm_fnv_bridge::npc::GeneratedLine;
use chasm_fnv_bridge::protocol::{NativeLocation, NativeRequest};
use chasm_fnv_bridge::sink::{OutgoingResponse, TurnSink, WrittenChunk};
use chasm_fnv_bridge::BridgeConfig;
use tokio::sync::mpsc::{self, UnboundedSender};

use crate::fnv_bridge::InProcessChasmClient;
use crate::nervous::{Decision, Stimulus};
use crate::radio::RadioFx;
use crate::{AppState, WebError, WebResult};

/// `POST /api/game/v1/turn` request body. The fields needed to build a
/// [`NativeRequest`] for one NPC turn. Text-only is the common path; an optional
/// base64 WAV (`audio_base64`) is transcribed up front for voice input.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct TurnRequest {
    /// Stable id for this turn (echoed on every event). Auto-generated if omitted.
    pub request_id: String,
    /// The NPC's native key (the `nativeNpcKey` used for mapping/voice).
    pub npc_key: String,
    /// The NPC's display name.
    pub npc_name: String,
    /// The player's typed words. Empty + `audio_base64` ⇒ transcribe first.
    pub player_text: String,
    /// Whether to synthesize TTS audio for the reply (default true).
    #[serde(default = "default_true")]
    pub want_tts: bool,
    /// Optional location fields (cell/worldspace/region/major/minor).
    pub location: TurnLocation,
    /// Free-form metadata coerced into the request's metadata map (distance,
    /// nearby NPCs, voice flags, …) — same keys the file request's line-10 blob
    /// carries.
    pub metadata: Map<String, Value>,
    /// Optional base64-encoded WAV for push-to-talk voice input. When present and
    /// `player_text` is empty, it is recognized (STT) and the transcript becomes
    /// the player text.
    pub audio_base64: String,
}

/// Location sub-object of [`TurnRequest`], mapped 1:1 onto [`NativeLocation`].
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct TurnLocation {
    pub cell: String,
    pub worldspace: String,
    pub region: String,
    pub major: String,
    pub minor: String,
}

fn default_true() -> bool {
    true
}

/// A [`TurnSink`] that serializes each turn output to an NDJSON line and pushes it
/// onto a channel the HTTP response body drains. The methods are synchronous (called
/// from inside the spawned `run_turn` task); sending is non-blocking, so audio chunks
/// stream out the instant they are produced — the whole point of the endpoint.
struct HttpStreamSink {
    tx: UnboundedSender<String>,
    /// Radio effect for this turn's audio when the NPC talks over comms.
    radio: Option<Mutex<RadioFx>>,
}

impl HttpStreamSink {
    fn new(tx: UnboundedSender<String>) -> Self {
        Self { tx, radio: None }
    }

    /// Route every audio chunk of this turn through the radio effect.
    fn with_radio(mut self, static_level: f32) -> Self {
        self.radio = Some(Mutex::new(RadioFx::new(static_level)));
        self
    }

    /// Serialize one event value to a `\n`-terminated NDJSON line and enqueue it.
    /// Returns `false` when the receiver is gone (the client hung up) — the caller
    /// decides whether that abandons the turn. A serialization failure only skips
    /// the one line and reports the channel as alive.
    fn emit(&self, event: Value) -> bool {
        match serde_json::to_string(&event) {
            Ok(line) => self.tx.send(format!("{line}\n")).is_ok(),
            Err(_) => true,
        }
    }
}

impl TurnSink for HttpStreamSink {
    fn reply(&self, request: &NativeRequest, resp: &OutgoingResponse) -> anyhow::Result<bool> {
        // The structured NPC reply — the same fields the response file carries,
        // as one JSON event. `audioFilename` is the first chunk's name (already
        // streamed inline as `audio.chunk`s); included for parity with the file path.
        self.emit(json!({
            "type": "reply",
            "requestId": request.request_id,
            "status": resp.status,
            "npcKey": resp.npc_key,
            "npcName": resp.npc_name,
            "text": resp.text,
            "audioFilename": resp.audio_filename,
            "error": resp.error,
            "playerText": resp.player_text,
            "gameMaster": {
                "action": resp.gm_action,
                "confidence": resp.gm_confidence,
                "shouldTrigger": resp.gm_should_trigger,
            },
        }));
        Ok(true)
    }

    fn audio_chunk(
        &self,
        request: &NativeRequest,
        line: &GeneratedLine,
        chunk: &AudioChunk,
        index: u32,
        extra_meta: &[String],
    ) -> anyhow::Result<WrittenChunk> {
        // The caption for this chunk: the chunk's own text, else the line text.
        let caption = chunk.text.clone().unwrap_or_else(|| line.text.clone());
        // Surface the incremental text as a `speech.delta` (mirrors the bridge's
        // chasm client) so a client can render the subtitle before/while the audio
        // plays, then the audio itself as an `audio.chunk`.
        if !caption.trim().is_empty() {
            let _ = self.emit(json!({
                "type": "speech.delta",
                "requestId": request.request_id,
                "index": index,
                "text": caption,
            }));
        }
        // Over the radio: band-limit + static (stateful across this turn's chunks).
        let radio_audio = self.radio.as_ref().and_then(|fx| {
            fx.lock()
                .unwrap_or_else(|e| e.into_inner())
                .process_wav(&chunk.audio)
        });
        let audio = radio_audio.as_deref().unwrap_or(&chunk.audio);
        let mut event = json!({
            "type": "audio.chunk",
            "requestId": request.request_id,
            "index": index,
            "audio": { "data": STANDARD.encode(audio) },
            "mimeType": if chunk.mime_type.is_empty() { "audio/wav" } else { &chunk.mime_type },
            "text": caption,
            "npcKey": line.native_npc_key,
            "npcName": line.native_npc_name,
        });
        if let Some(max) = chunk.caption_max_chars {
            event["captionMaxChars"] = json!(max);
        }
        // Forward any extra caption metadata (e.g. admin_voice / non_positional)
        // as a flat object, so the client gets the same flags the file chunk lines
        // carry without re-parsing `key=value` strings.
        if !extra_meta.is_empty() {
            event["metadata"] = meta_lines_to_object(extra_meta);
            // Voice direction cue ("whispering") — a bridge may also lower the
            // playback volume or pick an animation from it.
            if let Some(cue) = event["metadata"].get("delivery").cloned() {
                event["delivery"] = cue;
            }
        }
        if self.radio.is_some() {
            // Comms audio: play it in the player's ear, not from the NPC's body.
            event["channel"] = json!("radio");
            event["nonPositional"] = json!(true);
        }
        // Dead silence: when the client hung up mid-turn (the plugin abandons a reply
        // whose speaker died), abort the turn here instead of generating the rest of
        // the line into a dead channel. The error rides the existing sink plumbing —
        // it cancels the TTS stream and drops the LLM generation stream (which
        // cancels the runtime's slot), so no new protocol is involved.
        if !self.emit(event) {
            anyhow::bail!(
                "HTTP turn client disconnected (request {}); abandoning generation",
                request.request_id
            );
        }
        // The HTTP client plays the streamed bytes directly; there is no on-disk
        // file, but we still return a stable name + index for parity with FileSink.
        Ok(WrittenChunk {
            filename: chasm_fnv_bridge::sink::audio_filename(&request.request_id, index),
            index,
        })
    }

    fn action(
        &self,
        config: &BridgeConfig,
        request: &NativeRequest,
        actor: &ActionActor,
        gm: &GameMaster,
        source: &str,
    ) -> bool {
        // The HTTP transport does not (yet) own the durable control/actions queue —
        // that stays file-based for the C++ plugin to consume. We surface the
        // classified action so the client can act on it, and report `queued: false`
        // so run_turn leaves `shouldTrigger` armed on the reply (the HTTP client is
        // then the one that fires it, exactly once).
        let _ = (config, source);
        if !gm.should_trigger {
            return false;
        }
        // The chosen action's parameters (e.g. `target`), and — when the target
        // names something the NPC can see — that landmark with its coordinates.
        let parameters = gm
            .actions
            .iter()
            .find(|a| a.get("action_id").and_then(Value::as_str) == Some(gm.action_id.as_str()))
            .and_then(|a| a.get("parameters").cloned())
            .unwrap_or_else(|| json!({}));
        let target = if gm.target.trim().is_empty() {
            parameters.get("target").and_then(Value::as_str).unwrap_or("").to_string()
        } else {
            gm.target.clone()
        };
        let landmarks = crate::senses::landmarks_from_metadata(&Value::Object(
            request.metadata.clone().into_iter().collect(),
        ));
        let resolved_target = crate::senses::resolve_landmark_name(&target, &landmarks);
        self.emit(json!({
            "type": "action",
            "requestId": request.request_id,
            "action": gm.action,
            "confidence": gm.confidence,
            "shouldTrigger": gm.should_trigger,
            "actionId": gm.action_id,
            "reason": gm.reason,
            "target": target,
            "parameters": parameters,
            "resolvedTarget": resolved_target,
            "actor": {
                "npcKey": actor.native_npc_key,
                "npcName": actor.native_npc_name,
                "characterName": actor.character_name,
                "characterId": actor.character_id,
            },
            "queued": false,
        }));
        false
    }

    fn end_of_turn(&self) -> anyhow::Result<()> {
        self.emit(json!({ "type": "turn.completed" }));
        Ok(())
    }
}

/// Flatten the `key=value` caption metadata lines into a JSON object (string
/// values), for the `audio.chunk` event's `metadata` field.
fn meta_lines_to_object(lines: &[String]) -> Value {
    let mut map = Map::new();
    for line in lines {
        if let Some((key, value)) = line.split_once('=') {
            map.insert(key.trim().to_string(), json!(value.trim()));
        }
    }
    Value::Object(map)
}

/// Build the [`NativeRequest`] for this turn from the body, generating a request id
/// when absent and coercing the metadata map.
fn build_native_request(req: &TurnRequest) -> NativeRequest {
    let request_id = if req.request_id.trim().is_empty() {
        format!(
            "game_turn_{}",
            chasm_fnv_bridge::protocol::now_epoch_millis()
        )
    } else {
        req.request_id.trim().to_string()
    };
    NativeRequest {
        request_id,
        npc_key: req.npc_key.clone(),
        npc_name: req.npc_name.clone(),
        want_tts: req.want_tts,
        player_text: req.player_text.clone(),
        location: NativeLocation {
            cell: req.location.cell.clone(),
            worldspace: req.location.worldspace.clone(),
            region: req.location.region.clone(),
            major: req.location.major.clone(),
            minor: req.location.minor.clone(),
        },
        metadata: req.metadata.clone().into_iter().collect(),
    }
}

/// `POST /api/game/v1/turn`: run one NPC turn and stream its outputs as NDJSON.
///
/// Request: [`TurnRequest`] (npc_key, npc_name, player_text / audio_base64,
/// location, metadata, want_tts).
///
/// Response: `application/x-ndjson`, one JSON object per line, in production order:
/// `speech.delta` + `audio.chunk` pairs as the line is synthesized, then `action`
/// (if the turn chose one), then the structured `reply`, then a terminal
/// `turn.completed`. A fatal error before streaming starts is a non-200; an error
/// mid-turn is surfaced as a `turn.error` line.
pub async fn turn(
    State(state): State<Arc<AppState>>,
    Json(req): Json<TurnRequest>,
) -> WebResult<impl IntoResponse> {
    // Load the same bridge config the in-process fold uses, so NPC mapping, live-chat
    // identity, action books and TTS overrides match the file path exactly.
    let mut config = load_bridge_config(&state)?;
    let settings = AppSettings::load(&state.config.settings_path);

    let mut request = build_native_request(&req);

    // Voice input: transcribe the supplied WAV up front (via the in-process STT
    // handler) and use the transcript as the player text, so the turn proceeds on
    // the text path and never reaches the file-sidecar STT logic.
    if request.player_text.trim().is_empty() && !req.audio_base64.trim().is_empty() {
        let client = InProcessChasmClient::new(state.clone());
        let transcript = client.recognize(&recognize_body(&req.audio_base64)).await?;
        let transcript = transcript.trim().to_string();
        if transcript.is_empty() {
            return Err(WebError::from(anyhow::anyhow!(
                "speech recognition returned no text"
            )));
        }
        request.player_text = transcript;
    }

    // Senses: lines this NPC overheard since its last turn, and an optional
    // screenshot of its view (described once, never forwarded/persisted).
    let listener = turn_npc_name(&request);
    let metadata_value = Value::Object(request.metadata.clone().into_iter().collect());
    crate::senses::record_inline_heard(&metadata_value, &listener);
    request.metadata.remove("heard");
    if let Some(shot) = take_screenshot(&mut request) {
        if let Some(description) =
            crate::senses::describe_screenshot(&state, &settings, &listener, &shot).await
        {
            let vision = request
                .metadata
                .entry("vision".to_string())
                .or_insert_with(|| json!({}));
            if vision.is_object() {
                vision["scene_description"] = json!(description);
            }
        }
    }

    // Radio: a radio-capable NPC beyond the radio distance is heard over comms
    // instead of being out of earshot.
    let radio = radio_for_turn(&settings, &request);
    if radio {
        config.native_max_distance_meters = f64::MAX;
        request.metadata.insert("radio".to_string(), json!(true));
    }

    let (tx, rx) = mpsc::unbounded_channel::<String>();

    // Nervous system: react before the brain. A reflex streams to the bridge
    // immediately (before any LLM work); a bare command ends the turn there.
    if let Some(decision) = nervous_pass(&state, &settings, &request).await {
        let request_id = request.request_id.clone();
        let _ = tx.send(ndjson(&json!({
            "type": "nervous",
            "requestId": request_id,
            "npcName": listener,
            "decision": decision,
        })));
        if let Some(fired) = &decision.reflex {
            let _ = tx.send(ndjson(&json!({
                "type": "reflex",
                "requestId": request_id,
                "npcKey": request.npc_key,
                "npcName": listener,
                "action": fired.id,
                "description": fired.description,
                "target": fired.target,
                "confidence": decision.confidence,
                "source": decision.source,
                "latencyMs": decision.latency_ms,
            })));
        }
        let brain_skipped = decision.layer == "ignore"
            || (decision.layer == "reflex" && !decision.reply);
        if brain_skipped {
            let _ = tx.send(ndjson(&json!({
                "type": "reply",
                "requestId": request_id,
                "status": "1",
                "npcKey": request.npc_key,
                "npcName": listener,
                "text": "",
                "audioFilename": "",
                "error": "",
                "playerText": request.player_text,
                "gameMaster": {
                    "action": decision.reflex.as_ref().map(|r| r.id.as_str()).unwrap_or("NONE"),
                    "confidence": format!("{:.2}", decision.confidence),
                    // Already fired as the `reflex` event above.
                    "shouldTrigger": false,
                },
            })));
            let _ = tx.send(ndjson(&json!({ "type": "turn.completed" })));
            drop(tx);
            return Ok(ndjson_response(rx));
        }
        if let Some(note) = decision.brain_note() {
            request.metadata.insert(
                "nervous".to_string(),
                json!({
                    "note": note,
                    "layer": decision.layer,
                    "refused": decision.refused,
                    "reflex": decision.reflex.as_ref().map(|r| r.id.clone()),
                }),
            );
        }
    }

    // Drive the turn on a task, streaming each emitted line through the channel as
    // it is produced (so opener audio flows while the rest of the line generates).
    let task_state = state.clone();
    let radio_static = settings.voice_direction.radio_static;
    tokio::spawn(async move {
        let client: Arc<dyn ChasmClient> = Arc::new(InProcessChasmClient::new(task_state));
        let mut sink = HttpStreamSink::new(tx.clone());
        if radio {
            sink = sink.with_radio(radio_static);
        }
        if let Err(error) =
            chasm_fnv_bridge::run_turn_with_sink(&config, client.as_ref(), &sink, &request)
                .await
        {
            // Surface a mid-turn failure as a terminal error line (the pre-stream
            // failures already returned non-200 above).
            let line = serde_json::to_string(&json!({
                "type": "turn.error",
                "requestId": request.request_id,
                "error": error.to_string(),
            }))
            .unwrap_or_else(|_| "{\"type\":\"turn.error\"}".to_string());
            let _ = tx.send(format!("{line}\n"));
        }
    });

    Ok(ndjson_response(rx))
}

/// The streaming NDJSON response body draining `rx`.
fn ndjson_response(
    mut rx: mpsc::UnboundedReceiver<String>,
) -> ([(header::HeaderName, &'static str); 1], Body) {
    let body_stream = async_stream::stream! {
        while let Some(line) = rx.recv().await {
            yield Ok::<String, std::convert::Infallible>(line);
        }
    };
    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        Body::from_stream(body_stream),
    )
}

/// One `\n`-terminated NDJSON line.
fn ndjson(event: &Value) -> String {
    format!("{}\n", serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string()))
}

/// The NPC this turn is about: the request's NPC, else the crosshair / first
/// nearby NPC.
fn turn_npc_name(request: &NativeRequest) -> String {
    if !request.npc_name.trim().is_empty() {
        return request.npc_name.trim().to_string();
    }
    let nearby = nearby_list(request);
    nearby
        .iter()
        .find(|npc| npc.get("under_crosshair").and_then(Value::as_bool) == Some(true))
        .or_else(|| nearby.first())
        .and_then(|npc| npc.get("npc_name").or_else(|| npc.get("characterId")))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn nearby_list(request: &NativeRequest) -> Vec<Value> {
    request
        .metadata
        .get("targeting")
        .and_then(|t| t.get("nearby_npcs"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn metadata_flag(request: &NativeRequest, key: &str) -> bool {
    request.metadata.get(key).is_some_and(|v| {
        v.as_bool() == Some(true) || v.as_i64() == Some(1) || v.as_str() == Some("1")
    })
}

/// Removes `metadata.vision.screenshot_base64` (large; described, not stored).
fn take_screenshot(request: &mut NativeRequest) -> Option<String> {
    let vision = request.metadata.get_mut("vision")?.as_object_mut()?;
    let shot = vision.remove("screenshot_base64")?;
    shot.as_str().map(str::to_string).filter(|s| !s.trim().is_empty())
}

/// Whether this turn's NPC speaks over the radio: the bridge forced it
/// (`metadata.radio`), or the NPC is radio-capable (`metadata.radio_capable`)
/// and farther than the radio distance.
fn radio_for_turn(settings: &AppSettings, request: &NativeRequest) -> bool {
    let voice = &settings.voice_direction;
    if !voice.radio_enabled {
        return false;
    }
    if metadata_flag(request, "radio") {
        return true;
    }
    metadata_flag(request, "radio_capable")
        && chasm_fnv_bridge::npc::native_distance_meters(request) > f64::from(voice.radio_distance_m)
}

/// Runs the nervous system on the player's words, when it applies to this
/// turn: enabled, real player speech, and not an admin / reaction / synthetic
/// (world event, errand report) turn — those are already decided.
async fn nervous_pass(
    state: &AppState,
    settings: &AppSettings,
    request: &NativeRequest,
) -> Option<Decision> {
    if !settings.nervous_system.enabled
        || request.player_text.trim().is_empty()
        || metadata_flag(request, "nervous_skip")
        || chasm_fnv_bridge::admin::is_admin_request(request)
        || request.metadata.get("reaction").is_some_and(Value::is_object)
        || metadata_flag(request, "errand_completion")
        || metadata_flag(request, "world_event")
    {
        return None;
    }
    let metadata = Value::Object(request.metadata.clone().into_iter().collect());
    let nearby_npcs = nearby_list(request)
        .iter()
        .filter_map(|npc| npc.get("npc_name").and_then(Value::as_str).map(str::to_string))
        .collect();
    let mut situation: Vec<String> = Vec::new();
    let place = [&request.location.minor, &request.location.major]
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(", ");
    if !place.is_empty() {
        situation.push(format!("Location: {place}."));
    }
    if metadata_flag(request, "in_combat") {
        situation.push("In combat right now.".to_string());
    }
    if let Some(objective) = metadata.get("objective").and_then(Value::as_str) {
        situation.push(format!("Current objective: {objective}."));
    }
    let stimulus = Stimulus {
        npc_key: request.npc_key.clone(),
        npc_name: turn_npc_name(request),
        kind: "speech".to_string(),
        speaker: "player".to_string(),
        text: request.player_text.clone(),
        landmarks: crate::senses::landmarks_from_metadata(&metadata),
        nearby_npcs,
        situation: situation.join(" "),
        reflexes: None,
    };
    Some(crate::nervous::decide(state, settings, &stimulus).await)
}

/// Load the bridge config (the same `nvbridge.config.json` the in-process fold and
/// standalone bin read), resolving the path from launcher settings.
fn load_bridge_config(state: &AppState) -> WebResult<BridgeConfig> {
    let settings = AppSettings::load(&state.config.settings_path);
    // Blank helper config (the default for a fresh install) → `load_config` returns
    // built-in defaults pointed at the fixed rendezvous dir; no developer path.
    let config_path = settings.launcher.helper_config.trim().to_string();
    let mut config =
        chasm_fnv_bridge::load_config(std::path::Path::new(&config_path)).map_err(WebError::from)?;
    // With no explicit NPC map (fresh install), map nearby NPCs to the active
    // profile's characters by name so a turn resolves. Shared with the in-process bridge.
    crate::fnv_bridge::enrich_config_from_profile(&mut config, state);
    Ok(config)
}

/// The `/speech/recognize` body for a base64 WAV — the same shape `speech_recognize`
/// (and the bridge's file STT path) accepts.
fn recognize_body(audio_b64: &str) -> Value {
    json!({
        "audio": {
            "data": audio_b64,
            "encoding": "base64",
            "format": "wav",
            "mimeType": "audio/wav",
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures_util::Stream;
    use std::pin::Pin;

    // A canned ChasmClient: a fixed nearby NPC already resolves a participant (the
    // request carries an explicit `characterId`), so we only need to script the
    // streamed turn + one synthesized audio chunk. Everything the regular-NPC path
    // calls is answered minimally; the rest is unreachable for this turn.
    struct MockChasm {
        reply_text: String,
        audio: Vec<u8>,
    }

    #[async_trait]
    impl ChasmClient for MockChasm {
        async fn live_chat_exists(&self, _id: &str) -> anyhow::Result<bool> {
            Ok(true) // skip create_live_chat
        }
        async fn create_live_chat(&self, _body: &Value) -> anyhow::Result<()> {
            Ok(())
        }
        async fn presence(&self, _id: &str, _body: &Value) -> anyhow::Result<()> {
            Ok(())
        }
        async fn recognize(&self, _body: &Value) -> anyhow::Result<String> {
            Ok(String::new())
        }
        async fn generate_headless(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        async fn save_sync_event(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        async fn event_log_ingest(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        fn generate_stream_events<'a>(
            &'a self,
            _id: &str,
            _body: &Value,
        ) -> Pin<Box<dyn Stream<Item = anyhow::Result<Value>> + Send + 'a>> {
            let text = self.reply_text.clone();
            Box::pin(async_stream::try_stream! {
                yield json!({ "type": "speaker.start", "speaker": { "participantId": "npc:easy_pete" } });
                yield json!({ "type": "speech.delta", "text": text });
                yield json!({
                    "type": "live.completed",
                    "turn": { "message": { "content": text } },
                });
            })
        }
        async fn synthesize_stream(
            &self,
            _body: &Value,
            on_chunk: &mut (dyn FnMut(AudioChunk) -> anyhow::Result<()> + Send),
        ) -> anyhow::Result<usize> {
            on_chunk(AudioChunk {
                index: Some(0),
                audio: self.audio.clone(),
                mime_type: "audio/wav".into(),
                text: Some("Back so soon, wanderer?".into()),
                caption_max_chars: Some(80),
            })?;
            Ok(1)
        }
    }

    fn test_config() -> BridgeConfig {
        BridgeConfig {
            native_bridge_roots: Vec::new(),
            poll_ms: 100,
            api_base: "http://127.0.0.1:8000/api/headless/v1".into(),
            tts_api_base: String::new(),
            stt_api_base: String::new(),
            api_key: String::new(),
            request_timeout_ms: 180_000,
            live_chat_id: "fnv-goodsprings".into(),
            group_id: "fnv-goodsprings".into(),
            participant_id: "player".into(),
            character_id: "Easy Pete".into(),
            character_name: "Easy Pete".into(),
            npc_character_map: Map::new(),
            native_max_distance_meters: 10.0,
            gamestate_radius_meters: 30.0,
            enable_action_books: false,
            action_book_target_game: "fallout-new-vegas".into(),
            action_book_ids: vec!["Fallout New Vegas Action Book".into()],
            native_action_confidence: 0.65,
            model: String::new(),
            admin_character_id: "Todd".into(),
            admin_character_name: "Todd".into(),
            admin_action_book_limit: 12,
            admin_session_id: String::new(),
            tts: Value::Object(Map::new()),
            speech_recognition: Value::Object(Map::new()),
            speech_recognition_timeout_ms: 45_000,
            music_enabled: false,
        }
    }

    fn test_request() -> NativeRequest {
        let mut request = NativeRequest {
            request_id: "req_test_http".into(),
            npc_key: "easy_pete".into(),
            npc_name: "Easy Pete".into(),
            want_tts: true,
            player_text: "Hello there again.".into(),
            ..Default::default()
        };
        request.location.major = "Goodsprings".into();
        // One nearby NPC carrying an explicit characterId resolves a participant
        // with no npc_character_map needed.
        request.metadata.insert(
            "targeting".into(),
            json!({
                "nearby_npcs": [{
                    "npc_key": "easy_pete",
                    "npc_name": "Easy Pete",
                    "characterId": "Easy Pete",
                    "distance_m": 2.0,
                    "under_crosshair": true,
                }]
            }),
        );
        request
    }

    /// Drive a full turn through `run_turn_with_sink` + `HttpStreamSink` with a mock
    /// backend, and assert the streamed NDJSON carries the expected event sequence:
    /// a speech.delta, an audio.chunk (base64 WAV + index + caption), the structured
    /// reply, and a terminal turn.completed.
    #[tokio::test]
    async fn turn_streams_reply_audio_and_end() {
        let config = test_config();
        let request = test_request();
        let client = MockChasm {
            reply_text: "Back so soon, wanderer?".into(),
            audio: b"RIFFmock-wav-bytes".to_vec(),
        };

        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let sink = HttpStreamSink::new(tx);
        chasm_fnv_bridge::run_turn_with_sink(&config, &client, &sink, &request)
            .await
            .expect("turn should succeed");
        drop(sink); // drops the only sender, so draining terminates

        let mut events: Vec<Value> = Vec::new();
        while let Some(line) = rx.recv().await {
            for piece in line.split('\n').filter(|s| !s.trim().is_empty()) {
                events.push(serde_json::from_str(piece).expect("each line is valid JSON"));
            }
        }
        let types: Vec<&str> = events
            .iter()
            .map(|e| e.get("type").and_then(Value::as_str).unwrap_or(""))
            .collect();

        // Order: audio (with its speech.delta) streams first, then the reply, then end.
        assert!(types.contains(&"speech.delta"), "types: {types:?}");
        assert!(types.contains(&"audio.chunk"), "types: {types:?}");
        assert_eq!(types.last(), Some(&"turn.completed"), "types: {types:?}");
        let reply_pos = types.iter().position(|t| *t == "reply").expect("a reply event");
        let audio_pos = types.iter().position(|t| *t == "audio.chunk").unwrap();
        assert!(audio_pos < reply_pos, "audio should stream before the reply: {types:?}");

        let reply = events.iter().find(|e| e["type"] == "reply").unwrap();
        assert_eq!(reply["text"], "Back so soon, wanderer?");
        assert_eq!(reply["status"], "1");
        assert_eq!(reply["requestId"], "req_test_http");

        let chunk = events.iter().find(|e| e["type"] == "audio.chunk").unwrap();
        assert_eq!(chunk["index"], 0);
        assert_eq!(chunk["mimeType"], "audio/wav");
        assert_eq!(chunk["captionMaxChars"], 80);
        let decoded = STANDARD
            .decode(chunk["audio"]["data"].as_str().unwrap())
            .expect("audio.data is base64");
        assert_eq!(decoded, b"RIFFmock-wav-bytes");
    }

    /// Dead silence: when the HTTP client hangs up mid-turn (the plugin abandons a
    /// reply whose speaker died), the first audio chunk emitted into the dead channel
    /// aborts the turn — the remaining generation is cancelled instead of streaming
    /// into the void.
    #[tokio::test]
    async fn disconnected_client_aborts_turn() {
        let config = test_config();
        let request = test_request();
        let client = MockChasm {
            reply_text: "You'll never hear this line.".into(),
            audio: b"RIFFmock-wav-bytes".to_vec(),
        };

        let (tx, rx) = mpsc::unbounded_channel::<String>();
        drop(rx); // the client hung up before any output arrived
        let sink = HttpStreamSink::new(tx);
        let error = chasm_fnv_bridge::run_turn_with_sink(&config, &client, &sink, &request)
            .await
            .expect_err("a dead channel should abort the turn at the first audio chunk");
        assert!(
            error.to_string().contains("disconnected"),
            "unexpected error: {error:#}"
        );
    }

    /// The `action` sink event: a triggering game_master is surfaced as an `action`
    /// line, and `action()` returns false so run_turn keeps `shouldTrigger` armed on
    /// the reply (the HTTP client fires it, since the durable queue is file-only).
    #[tokio::test]
    async fn action_event_is_emitted_and_unqueued() {
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let sink = HttpStreamSink::new(tx);
        let config = test_config();
        let request = test_request();
        let actor = ActionActor {
            native_npc_key: "sony".into(),
            native_npc_name: "Sony".into(),
            character_name: "Sony".into(),
            character_id: "Sony".into(),
        };
        let gm = GameMaster {
            action: "ATTACK".into(),
            confidence: "0.92".into(),
            should_trigger: true,
            action_id: "attack-1".into(),
            reason: "player asked".into(),
            actions: Vec::new(),
            ..GameMaster::none()
        };
        let queued = sink.action(&config, &request, &actor, &gm, "test-source");
        assert!(!queued, "HTTP sink never owns the durable queue");
        drop(sink); // close the channel so draining terminates

        let line = rx.recv().await.expect("an action line");
        let event: Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(event["type"], "action");
        assert_eq!(event["action"], "ATTACK");
        assert_eq!(event["confidence"], "0.92");
        assert_eq!(event["shouldTrigger"], true);
        assert_eq!(event["queued"], false);
        assert_eq!(event["actor"]["npcKey"], "sony");
    }

    /// A mock that streams a voice-directed line and records each TTS body.
    struct DirectedMock {
        line: String,
        synth_bodies: std::sync::Mutex<Vec<Value>>,
    }

    #[async_trait]
    impl ChasmClient for DirectedMock {
        async fn live_chat_exists(&self, _id: &str) -> anyhow::Result<bool> {
            Ok(true)
        }
        async fn create_live_chat(&self, _body: &Value) -> anyhow::Result<()> {
            Ok(())
        }
        async fn presence(&self, _id: &str, _body: &Value) -> anyhow::Result<()> {
            Ok(())
        }
        async fn recognize(&self, _body: &Value) -> anyhow::Result<String> {
            Ok(String::new())
        }
        async fn generate_headless(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        async fn save_sync_event(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        async fn event_log_ingest(&self, _body: &Value) -> anyhow::Result<Value> {
            Ok(Value::Null)
        }
        fn generate_stream_events<'a>(
            &'a self,
            _id: &str,
            _body: &Value,
        ) -> Pin<Box<dyn Stream<Item = anyhow::Result<Value>> + Send + 'a>> {
            let text = self.line.clone();
            Box::pin(async_stream::try_stream! {
                yield json!({ "type": "speaker.start", "speaker": { "participantId": "npc:easy_pete" } });
                yield json!({ "type": "speech.delta", "text": text });
                yield json!({ "type": "live.completed", "turn": { "message": { "content": text } } });
            })
        }
        async fn synthesize_stream(
            &self,
            body: &Value,
            on_chunk: &mut (dyn FnMut(AudioChunk) -> anyhow::Result<()> + Send),
        ) -> anyhow::Result<usize> {
            self.synth_bodies.lock().unwrap().push(body.clone());
            on_chunk(AudioChunk {
                index: Some(0),
                audio: b"RIFFmock".to_vec(),
                mime_type: "audio/wav".into(),
                text: None,
                caption_max_chars: None,
            })?;
            Ok(1)
        }
    }

    /// Voice direction end to end through run_turn: the leading cue becomes the
    /// TTS `instructions`, vocal events stay in the spoken text, and neither
    /// reaches the caption, the chunk event, or the structured reply.
    #[tokio::test]
    async fn voice_direction_cue_reaches_tts_not_captions() {
        let client = DirectedMock {
            line: "[whispering] Grunt at two o'clock. (sigh) Stay low.".into(),
            synth_bodies: Default::default(),
        };
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let sink = HttpStreamSink::new(tx);
        chasm_fnv_bridge::run_turn_with_sink(&test_config(), &client, &sink, &test_request())
            .await
            .expect("turn should succeed");
        drop(sink);

        let bodies = client.synth_bodies.lock().unwrap().clone();
        assert_eq!(bodies.len(), 1);
        assert_eq!(bodies[0]["instructions"], "whispering");
        assert_eq!(bodies[0]["text"], "Grunt at two o'clock. (sigh) Stay low.");

        let mut events: Vec<Value> = Vec::new();
        while let Some(line) = rx.recv().await {
            events.push(serde_json::from_str(line.trim()).unwrap());
        }
        let chunk = events.iter().find(|e| e["type"] == "audio.chunk").unwrap();
        assert_eq!(chunk["delivery"], "whispering");
        assert_eq!(chunk["text"], "Grunt at two o'clock. Stay low.");
        let reply = events.iter().find(|e| e["type"] == "reply").unwrap();
        assert_eq!(reply["text"], "Grunt at two o'clock. Stay low.");
    }

    /// Radio: chunks are run through the comms effect and flagged for 2D playback.
    #[tokio::test]
    async fn radio_sink_filters_audio_and_flags_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let sink = HttpStreamSink::new(tx).with_radio(0.05);
        let pcm: Vec<u8> = (0..2400i32)
            .flat_map(|i| (((i as f32 * 0.26).sin() * 8000.0) as i16).to_le_bytes())
            .collect();
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&24_000u32.to_le_bytes());
        wav.extend_from_slice(&48_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        wav.extend_from_slice(&pcm);
        let line = GeneratedLine {
            participant_id: "npc:kevin".into(),
            native_npc_key: "kevin".into(),
            native_npc_name: "Kevin".into(),
            character_name: "Kevin".into(),
            character_id: "Kevin".into(),
            text: "Contact, left side.".into(),
            turn: Value::Null,
        };
        let chunk = AudioChunk {
            index: Some(0),
            audio: wav.clone(),
            mime_type: "audio/wav".into(),
            text: None,
            caption_max_chars: None,
        };
        sink.audio_chunk(&test_request(), &line, &chunk, 0, &[]).unwrap();
        drop(sink);
        let mut events: Vec<Value> = Vec::new();
        while let Some(line) = rx.recv().await {
            events.push(serde_json::from_str(line.trim()).unwrap());
        }
        let audio = events.iter().find(|e| e["type"] == "audio.chunk").unwrap();
        assert_eq!(audio["channel"], "radio");
        assert_eq!(audio["nonPositional"], true);
        let bytes = STANDARD.decode(audio["audio"]["data"].as_str().unwrap()).unwrap();
        assert!(bytes.starts_with(b"RIFF"));
        assert!(bytes.len() > wav.len(), "squelch burst prepended on the first chunk");
        assert_ne!(&bytes[44..44 + 200], &wav[44..44 + 200]);
    }

    #[test]
    fn radio_turns_need_capability_and_distance() {
        let settings = AppSettings::default();
        let mut request = test_request();
        assert!(!radio_for_turn(&settings, &request), "2m away, not radio-capable");
        request.metadata.insert("radio_capable".into(), json!(true));
        assert!(!radio_for_turn(&settings, &request), "radio-capable but close");
        request.metadata.insert(
            "targeting".into(),
            json!({ "nearby_npcs": [{ "npc_key": "easy_pete", "npc_name": "Easy Pete", "distance_m": 60.0 }] }),
        );
        assert!(radio_for_turn(&settings, &request), "radio-capable and far");
        let mut forced = test_request();
        forced.metadata.insert("radio".into(), json!("1"));
        assert!(radio_for_turn(&settings, &forced));
    }

}
