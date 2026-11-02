//! NPC senses: what a character can **see** and **hear** right now.
//!
//! Both are bridge-driven and game-agnostic:
//!
//! * **Sight.** The bridge reports the notable things in the NPC's view as
//!   [`Landmark`]s (`metadata.vision.landmarks` on a turn, or the `/look` body):
//!   a label, a world position, and how far/where it is. They are injected as a
//!   depth-1 "what you can see" block so the brain can talk about them, and the
//!   nervous system resolves spoken references ("stand by that rock") to one of
//!   them so a `move_to` reflex carries real coordinates. Optionally the bridge
//!   also sends a screenshot, which a vision-capable model (Gemma 4 with its
//!   mmproj, or any OpenAI-compatible vision endpoint) turns into a short scene
//!   description.
//! * **Hearing.** In-game voice lines the NPC was in earshot of — other NPCs,
//!   radio chatter — arrive via `POST /api/game/v1/hear` (or inline as
//!   `metadata.heard` on a turn) and are kept in a short rolling window. The
//!   newest lines ride the same depth-1 block, so a squadmate can react to
//!   "they're trying to flank us" without the player repeating it.
//!
//! The hearing store is in-memory and short-lived by design: it is ambient
//! context, not memory. Durable, save-aware memory stays with the event log and
//! witness system.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use axum::{extract::State, Json};
use chasm_core::{AppSettings, PromptComponentView, SensesSettings};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{AppState, WebResult};

/// Upper bound on remembered lines across all listeners (oldest are dropped).
const HEARING_CAPACITY: usize = 256;
/// Upper bound on landmarks rendered into one prompt.
const MAX_PROMPT_LANDMARKS: usize = 12;

/// A visible thing the bridge reported: a rock, a pile of bodies, a Warthog.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Landmark {
    /// Bridge-side stable id (handed back on a resolved `move_to` target).
    pub id: String,
    /// What it is, in plain words ("large boulder", "pile of Marine bodies").
    pub label: String,
    /// Optional category ("cover", "vehicle", "corpse", "enemy", …).
    pub kind: String,
    /// World-space position, in game units.
    pub position: Option<Position>,
    /// Distance from the NPC (meters), when known.
    pub distance_m: Option<f64>,
    /// Coarse direction relative to the NPC ("ahead", "left", "behind-right").
    pub direction: String,
}

/// A world-space position in game units.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Landmark {
    /// One human-readable line for the prompt / the decision model's state.
    pub fn describe(&self) -> String {
        let mut out = self.label.trim().to_string();
        let mut details: Vec<String> = Vec::new();
        if let Some(distance) = self.distance_m {
            details.push(format!("{distance:.0}m"));
        }
        if !self.direction.trim().is_empty() {
            details.push(self.direction.trim().to_string());
        }
        if !details.is_empty() {
            out.push_str(&format!(" ({})", details.join(", ")));
        }
        out
    }
}

/// Reads `metadata.vision.landmarks` (also accepts a top-level `landmarks`).
pub fn landmarks_from_metadata(metadata: &Value) -> Vec<Landmark> {
    let list = metadata
        .pointer("/vision/landmarks")
        .or_else(|| metadata.get("landmarks"))
        .and_then(Value::as_array);
    let Some(list) = list else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|item| serde_json::from_value::<Landmark>(item.clone()).ok())
        .filter(|landmark| !landmark.label.trim().is_empty())
        .collect()
}

/// Finds the landmark a brain-chosen action named (`move_to(target="large
/// boulder")`): an exact label/id match, else the best word overlap.
pub fn resolve_landmark_name(name: &str, landmarks: &[Landmark]) -> Option<Landmark> {
    let wanted = name.trim();
    if wanted.is_empty() || landmarks.is_empty() {
        return None;
    }
    landmarks
        .iter()
        .find(|l| {
            l.label.trim().eq_ignore_ascii_case(wanted) || l.id.trim().eq_ignore_ascii_case(wanted)
        })
        .cloned()
        .or_else(|| crate::nervous::match_landmark_text(wanted, landmarks))
}

// ---------------------------------------------------------------------------
// Hearing
// ---------------------------------------------------------------------------

/// One overheard voice line.
#[derive(Debug, Clone)]
struct HeardLine {
    at: Instant,
    speaker: String,
    text: String,
    /// `world` (heard in the air) or `radio` (comms / an AI in your ear).
    channel: String,
    /// Lowercased names of who heard it; empty = everyone in earshot.
    listeners: Vec<String>,
}

fn hearing_store() -> &'static Mutex<VecDeque<HeardLine>> {
    static STORE: OnceLock<Mutex<VecDeque<HeardLine>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(VecDeque::new()))
}

/// Remembers one overheard line. Blank text is ignored.
pub fn record_heard(speaker: &str, text: &str, channel: &str, listeners: &[String]) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let channel = match channel.trim().to_ascii_lowercase().as_str() {
        "radio" | "comms" | "intercom" => "radio",
        _ => "world",
    };
    let line = HeardLine {
        at: Instant::now(),
        speaker: speaker.trim().to_string(),
        text: text.to_string(),
        channel: channel.to_string(),
        listeners: listeners
            .iter()
            .map(|name| name.trim().to_lowercase())
            .filter(|name| !name.is_empty())
            .collect(),
    };
    let mut store = hearing_store().lock().unwrap_or_else(|e| e.into_inner());
    store.push_back(line);
    while store.len() > HEARING_CAPACITY {
        store.pop_front();
    }
}

/// The lines `listener` heard within the window, oldest first, excluding their
/// own voice, capped to the newest `max_lines`.
fn heard_by(listener: &str, window: Duration, max_lines: usize) -> Vec<HeardLine> {
    let listener = listener.trim().to_lowercase();
    let store = hearing_store().lock().unwrap_or_else(|e| e.into_inner());
    let mut lines: Vec<HeardLine> = store
        .iter()
        .filter(|line| line.at.elapsed() <= window)
        .filter(|line| line.speaker.to_lowercase() != listener)
        .filter(|line| line.listeners.is_empty() || line.listeners.contains(&listener))
        .cloned()
        .collect();
    if lines.len() > max_lines {
        lines.drain(..lines.len() - max_lines);
    }
    lines
}

/// Plain-text lines the listener recently heard (for the decision model's state).
pub fn recent_heard_text(settings: &SensesSettings, listener: &str) -> Vec<String> {
    if !settings.hearing_enabled {
        return Vec::new();
    }
    heard_by(
        listener,
        Duration::from_secs(settings.hearing_window_secs.max(1)),
        settings.hearing_max_lines.max(1),
    )
    .into_iter()
    .map(|line| format_heard(&line))
    .collect()
}

fn format_heard(line: &HeardLine) -> String {
    let ago = line.at.elapsed().as_secs();
    let speaker = if line.speaker.is_empty() {
        "Someone"
    } else {
        &line.speaker
    };
    let channel = if line.channel == "radio" {
        " [radio]"
    } else {
        ""
    };
    format!("{speaker}{channel}: \"{}\" ({ago}s ago)", line.text)
}

/// Records the `metadata.heard` lines a turn carried inline (heard by `listener`).
pub fn record_inline_heard(metadata: &Value, listener: &str) {
    let Some(list) = metadata.get("heard").and_then(Value::as_array) else {
        return;
    };
    let listeners = if listener.trim().is_empty() {
        Vec::new()
    } else {
        vec![listener.to_string()]
    };
    for item in list {
        let field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or("");
        record_heard(
            field("speaker"),
            field("text"),
            field("channel"),
            &listeners,
        );
    }
}

/// `POST /api/game/v1/hear` body: one overheard line (or a batch in `lines`).
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HearRequest {
    pub speaker: String,
    pub text: String,
    /// `world` (default) or `radio`.
    pub channel: String,
    /// Character names that heard it; empty = everyone in earshot.
    pub listeners: Vec<String>,
    /// Several lines at once (each may carry its own speaker/text/channel).
    pub lines: Vec<HearLine>,
    /// Also run each listener's nervous system on the line and return its
    /// decision (e.g. a reflex to take cover when someone yells "grenade!").
    pub react: bool,
    /// Visible landmarks for the reaction pass (optional).
    pub landmarks: Vec<Landmark>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct HearLine {
    pub speaker: String,
    pub text: String,
    pub channel: String,
}

/// `POST /api/game/v1/hear`: remember overheard dialogue; optionally react.
pub async fn hear(
    State(state): State<Arc<AppState>>,
    Json(req): Json<HearRequest>,
) -> WebResult<Json<Value>> {
    let mut lines: Vec<HearLine> = req.lines;
    if !req.text.trim().is_empty() {
        lines.insert(
            0,
            HearLine {
                speaker: req.speaker.clone(),
                text: req.text.clone(),
                channel: req.channel.clone(),
            },
        );
    }
    for line in &lines {
        let channel = if line.channel.is_empty() {
            &req.channel
        } else {
            &line.channel
        };
        record_heard(&line.speaker, &line.text, channel, &req.listeners);
    }

    let mut reactions: Vec<Value> = Vec::new();
    if req.react && !req.listeners.is_empty() {
        let settings = AppSettings::load(&state.config.settings_path);
        for listener in &req.listeners {
            for line in &lines {
                if line.speaker.trim().eq_ignore_ascii_case(listener.trim()) {
                    continue;
                }
                let stimulus = crate::nervous::Stimulus {
                    npc_name: listener.clone(),
                    kind: "heard".to_string(),
                    speaker: line.speaker.clone(),
                    text: line.text.clone(),
                    landmarks: req.landmarks.clone(),
                    ..Default::default()
                };
                let decision = crate::nervous::decide(&state, &settings, &stimulus).await;
                reactions
                    .push(json!({ "listener": listener, "line": line.text, "decision": decision }));
            }
        }
    }
    Ok(Json(
        json!({ "ok": true, "recorded": lines.len(), "reactions": reactions }),
    ))
}

// ---------------------------------------------------------------------------
// Sight: screenshot description + `/look`
// ---------------------------------------------------------------------------

/// Asks a vision-capable, OpenAI-compatible model for a short description of
/// what the NPC sees in `screenshot_b64` (PNG or JPEG). `None` when screenshot
/// vision is off, the image is empty, or the call fails — sight then degrades
/// to the landmark list alone.
pub async fn describe_screenshot(
    state: &AppState,
    settings: &AppSettings,
    npc_name: &str,
    screenshot_b64: &str,
) -> Option<String> {
    let senses = &settings.senses;
    let image = screenshot_b64.trim();
    if !senses.screenshot_enabled || image.is_empty() {
        return None;
    }
    let base = if senses.vision_base_url.trim().is_empty() {
        format!("{}/v1", state.config.llm_endpoint.trim_end_matches('/'))
    } else {
        senses
            .vision_base_url
            .trim()
            .trim_end_matches('/')
            .to_string()
    };
    let data_url = if image.starts_with("data:") {
        image.to_string()
    } else {
        let mime = if image.starts_with("/9j/") {
            "image/jpeg"
        } else {
            "image/png"
        };
        format!("data:{mime};base64,{image}")
    };
    let who = if npc_name.trim().is_empty() {
        "the character"
    } else {
        npc_name.trim()
    };
    let mut body = json!({
        "messages": [{
            "role": "user",
            "content": [
                { "type": "text", "text": format!(
                    "This is what {who} sees right now in a video game, from their eyes. \
                     In at most three short sentences, describe the notable things in view \
                     (terrain, cover, people, vehicles, enemies, bodies) and where they are \
                     (left/right/ahead, near/far). Plain description only."
                ) },
                { "type": "image_url", "image_url": { "url": data_url } },
            ],
        }],
        "max_tokens": 160,
        "temperature": 0.2,
        "stream": false,
    });
    if !senses.vision_model.trim().is_empty() {
        body["model"] = json!(senses.vision_model.trim());
    }
    let mut request = vision_client()
        .post(format!("{base}/chat/completions"))
        .json(&body);
    if !senses.vision_api_key.trim().is_empty() {
        request = request.bearer_auth(senses.vision_api_key.trim());
    }
    let response = match request.send().await {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            tracing::warn!("vision: describe failed with status {}", response.status());
            return None;
        }
        Err(error) => {
            tracing::warn!("vision: describe request failed: {error}");
            return None;
        }
    };
    let value: Value = response.json().await.ok()?;
    let text = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    (!text.is_empty()).then_some(text)
}

fn vision_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap_or_default()
    })
}

/// `POST /api/game/v1/look` body.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct LookRequest {
    pub npc_name: String,
    /// What was said ("go stand near those bodies"). Empty = just describe.
    pub text: String,
    pub landmarks: Vec<Landmark>,
    /// Optional base64 PNG/JPEG of the NPC's view.
    pub screenshot_base64: String,
}

/// `POST /api/game/v1/look`: resolve a spoken reference to a visible landmark
/// (and, with a screenshot, describe the view).
pub async fn look(
    State(state): State<Arc<AppState>>,
    Json(req): Json<LookRequest>,
) -> WebResult<Json<Value>> {
    let settings = AppSettings::load(&state.config.settings_path);
    let description =
        describe_screenshot(&state, &settings, &req.npc_name, &req.screenshot_base64).await;
    let resolved = if req.text.trim().is_empty() || req.landmarks.is_empty() {
        None
    } else {
        Some(
            crate::nervous::ground_reference(&settings, &req.npc_name, &req.text, &req.landmarks)
                .await,
        )
    };
    Ok(Json(json!({
        "target": resolved.as_ref().and_then(|r| r.target.clone()),
        "confidence": resolved.as_ref().map(|r| r.confidence),
        "source": resolved.as_ref().map(|r| r.source.clone()),
        "description": description,
    })))
}

// ---------------------------------------------------------------------------
// Prompt injection
// ---------------------------------------------------------------------------

/// A boolean feature flag from the active profile's `profile.json`
/// (`features.<key>`), so a game profile can switch on what it needs.
pub fn profile_feature(state: &AppState, settings: &AppSettings, key: &str) -> bool {
    chasm_core::GameProfile::read(&state.config.profiles_dir, &settings.profile)
        .and_then(|profile| {
            profile
                .extra
                .get("features")
                .and_then(|f| f.get(key))
                .cloned()
        })
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

/// Voice direction is on when Settings say so or the profile asks for it.
pub fn voice_direction_active(state: &AppState, settings: &AppSettings) -> bool {
    settings.voice_direction.enabled || profile_feature(state, settings, "voiceDirection")
}

/// Builds one depth-1 system component (same shape as the combat alert).
pub fn component(key: &str, label: &str, note: &str, content: String) -> PromptComponentView {
    let char_count = content.chars().count();
    PromptComponentView {
        order: 0,
        group: "system".to_string(),
        key: key.to_string(),
        label: label.to_string(),
        role: "system".to_string(),
        status: "included".to_string(),
        note: note.to_string(),
        content,
        char_count,
    }
}

/// The per-turn senses blocks for `speaker_name`: what they can see (bridge
/// landmarks + optional screenshot description) and what they recently heard.
/// Empty when the bridge sent nothing — byte-identical prompts for games that
/// don't use senses.
pub fn prompt_components(
    settings: &SensesSettings,
    speaker_name: &str,
    player_metadata: &Value,
) -> Vec<PromptComponentView> {
    let mut out = Vec::new();
    if let Some(objective) = player_metadata
        .get("objective")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        out.push(component(
            "mission",
            "Mission",
            "Depth-1: the current mission objective reported by the game.",
            format!("[Current mission objective: {objective}]"),
        ));
    }
    if settings.vision_enabled {
        let landmarks = landmarks_from_metadata(player_metadata);
        let scene = player_metadata
            .pointer("/vision/scene_description")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty());
        if !landmarks.is_empty() || scene.is_some() {
            let mut text = String::from("[What you can see right now");
            text.push_str(":]\n");
            if let Some(scene) = scene {
                text.push_str(scene);
                text.push('\n');
            }
            for landmark in landmarks.iter().take(MAX_PROMPT_LANDMARKS) {
                text.push_str(&format!("- {}\n", landmark.describe()));
            }
            text.push_str(
                "You can refer to these naturally, and walk to one when asked (use its name as the target).",
            );
            out.push(component(
                "vision",
                "Vision",
                "Depth-1: bridge-reported landmarks (+ screenshot description) in the NPC's view.",
                text,
            ));
        }
    }
    if settings.hearing_enabled {
        let heard = recent_heard_text(settings, speaker_name);
        if !heard.is_empty() {
            let mut text = String::from(
                "[Voices you recently overheard nearby or over the radio (not said to you unless your name is used):]\n",
            );
            for line in heard {
                text.push_str(&format!("- {line}\n"));
            }
            out.push(component(
                "overheard",
                "Overheard",
                "Depth-1: in-game dialogue / radio chatter this NPC heard recently.",
                text.trim_end().to_string(),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn landmarks_parse_and_describe() {
        let metadata = json!({
            "vision": { "landmarks": [
                { "id": "rock_1", "label": "large boulder", "distance_m": 12.4, "direction": "ahead-left",
                  "position": { "x": 10.0, "y": 2.0, "z": 0.5 } },
                { "id": "blank", "label": "  " }
            ]}
        });
        let landmarks = landmarks_from_metadata(&metadata);
        assert_eq!(landmarks.len(), 1);
        assert_eq!(landmarks[0].describe(), "large boulder (12m, ahead-left)");
        assert_eq!(landmarks[0].position.unwrap().x, 10.0);
    }

    #[test]
    fn hearing_is_scoped_to_listeners_and_excludes_own_voice() {
        let tag = format!("t{}", std::process::id());
        let alice = format!("Alice{tag}");
        let bob = format!("Bob{tag}");
        // The store is process-global and tests run in parallel: scope every
        // line to this test's own listeners so no other test's prompt sees it.
        let both = [alice.clone(), bob.clone()];
        record_heard("Cortana", "They're trying to flank us.", "radio", &both);
        record_heard(&alice, "Only Bob hears this.", "world", &[bob.clone()]);
        record_heard(&bob, "Bob talking.", "world", &both);
        let senses = SensesSettings::default();
        let bob_heard = recent_heard_text(&senses, &bob);
        assert!(bob_heard.iter().any(|l| l.contains("Cortana [radio]")));
        assert!(bob_heard.iter().any(|l| l.contains("Only Bob hears this")));
        assert!(!bob_heard.iter().any(|l| l.contains("Bob talking")));
        let alice_heard = recent_heard_text(&senses, &alice);
        assert!(alice_heard.iter().any(|l| l.contains("Cortana [radio]")));
        assert!(!alice_heard
            .iter()
            .any(|l| l.contains("Only Bob hears this")));
    }

    #[test]
    fn components_empty_without_senses_data() {
        let senses = SensesSettings {
            hearing_enabled: false,
            ..SensesSettings::default()
        };
        assert!(prompt_components(&senses, "Nobody", &json!({})).is_empty());
        let with_view = prompt_components(
            &senses,
            "Nobody",
            &json!({ "vision": { "landmarks": [{ "label": "Warthog", "distance_m": 4 }] } }),
        );
        assert_eq!(with_view.len(), 1);
        assert!(with_view[0].content.contains("- Warthog (4m)"));
    }
}
