//! The **nervous system**: a fast, non-generative decision layer that sees every
//! stimulus before the LLM "brain" does.
//!
//! A real person doesn't deliberate before ducking. When the player says
//! "Kevin, jump", the nervous system decides in tens of milliseconds that this
//! is a plain physical command, fires the `jump` reflex straight to the bridge,
//! and only wakes the brain if the character would also say something about it.
//! "Can *you* jump?" is a question, so it goes to the brain. A command the
//! character would rather not follow is **refused**: no reflex fires and the
//! brain is told they decided not to, so the refusal is voiced in character.
//!
//! Two providers, chosen in Settings (`nervous_system.provider`):
//!
//! * **`kev`** — a Kev decision model (e.g. `jaredpalmer/kev-4b`) behind its
//!   System One endpoint (`POST {kev_url}/v1/systemone`). One request asks a set
//!   of typed questions about the moment (route, which reflex, would they
//!   comply, should they speak, which visible thing is meant) and gets
//!   calibrated probabilities back in a single prefill pass — no generation.
//! * **`rules`** — a built-in imperative matcher over the reflex catalog's
//!   phrases. No model, always available, and the automatic fallback when Kev
//!   is unreachable or slow.
//!
//! The reflex catalog is game-specific and comes from the active profile's
//! `profile.json` (`reflexes`), falling back to a small generic set. The bridge
//! executes reflexes by id; `move_to` style reflexes carry a resolved
//! [`Landmark`] (with coordinates) from the senses module.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use axum::{extract::State, Json};
use chasm_core::{AppSettings, GameProfile, NervousSystemSettings};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::senses::Landmark;
use crate::{AppState, WebResult};

/// One reflex the body can perform without thinking.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Reflex {
    /// Bridge action id ("jump", "move_to", …).
    pub id: String,
    /// What it does, for the decision model ("Jump in place").
    pub description: String,
    /// Imperative phrases that trigger it for the rules matcher ("get down").
    pub phrases: Vec<String>,
    /// Whether it needs a visible target (resolved to a [`Landmark`]).
    #[serde(alias = "needsTarget")]
    pub needs_target: bool,
}

fn reflex(id: &str, description: &str, phrases: &[&str], needs_target: bool) -> Reflex {
    Reflex {
        id: id.to_string(),
        description: description.to_string(),
        phrases: phrases.iter().map(|p| p.to_string()).collect(),
        needs_target,
    }
}

/// The generic reflex set used when the active profile defines none.
pub fn default_reflexes() -> Vec<Reflex> {
    vec![
        reflex("jump", "Jump in place", &["jump", "hop"], false),
        reflex(
            "crouch",
            "Crouch down low",
            &["crouch", "get down", "duck", "get low", "stay low"],
            false,
        ),
        reflex(
            "stand",
            "Stand back up from crouching",
            &["stand up", "get up", "on your feet", "stand"],
            false,
        ),
        reflex(
            "follow",
            "Follow the speaker",
            &["follow me", "on me", "with me", "come with me", "come on"],
            false,
        ),
        reflex(
            "wait",
            "Stop and hold position right here",
            &[
                "wait here",
                "stay here",
                "hold position",
                "hold here",
                "wait",
                "stay",
            ],
            false,
        ),
        reflex(
            "hold_fire",
            "Stop shooting and hold fire",
            &[
                "hold fire",
                "hold your fire",
                "cease fire",
                "stop shooting",
                "don't shoot",
                "do not shoot",
            ],
            false,
        ),
        reflex(
            "fire_at_will",
            "Open fire on enemies right now",
            &[
                "open fire",
                "fire at will",
                "weapons free",
                "shoot",
                "fire",
                "take them out",
                "light them up",
            ],
            false,
        ),
        reflex(
            "take_cover",
            "Get into the nearest cover",
            &["take cover", "get to cover", "find cover", "get in cover"],
            false,
        ),
        reflex(
            "move_to",
            "Walk over to a specific visible thing or spot",
            &[
                "go stand by",
                "go stand near",
                "go stand next to",
                "stand by",
                "stand near",
                "stand next to",
                "go to",
                "move to",
                "go over to",
                "get over to",
                "head to",
                "head over to",
                "walk to",
                "run to",
                "go near",
                "get to",
                "go by",
            ],
            true,
        ),
        reflex(
            "enter_vehicle",
            "Get into the nearest vehicle",
            &[
                "get in the car",
                "get in the vehicle",
                "get in",
                "hop in",
                "mount up",
            ],
            false,
        ),
        reflex(
            "exit_vehicle",
            "Get out of the vehicle",
            &["get out of the car", "get out", "hop out", "dismount"],
            false,
        ),
        reflex("reload", "Reload the weapon", &["reload"], false),
        reflex(
            "throw_grenade",
            "Throw a grenade at the enemy",
            &[
                "throw a grenade",
                "throw a nade",
                "nade them",
                "frag them",
                "grenade",
            ],
            false,
        ),
    ]
}

/// The active profile's reflex catalog (`profile.json` → `reflexes`), else the
/// generic set.
pub fn profile_reflexes(state: &AppState, settings: &AppSettings) -> Vec<Reflex> {
    GameProfile::read(&state.config.profiles_dir, &settings.profile)
        .and_then(|profile| profile.extra.get("reflexes").cloned())
        .and_then(|value| serde_json::from_value::<Vec<Reflex>>(value).ok())
        .filter(|list| !list.is_empty())
        .unwrap_or_else(default_reflexes)
}

/// Something the NPC perceived that may need a reaction.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Stimulus {
    pub npc_key: String,
    /// Who is reacting (their character name).
    pub npc_name: String,
    /// `speech` (someone talked, default), `heard` (overheard line) or `event`.
    pub kind: String,
    /// Who produced it ("player" for the player's voice).
    pub speaker: String,
    /// What was said / what happened.
    pub text: String,
    /// Visible landmarks (for `move_to` targets).
    pub landmarks: Vec<Landmark>,
    /// Names of other characters present (to tell who is being addressed).
    pub nearby_npcs: Vec<String>,
    /// Short free-text situation (location, combat, vehicle, …).
    pub situation: String,
    /// Per-request reflex catalog override.
    pub reflexes: Option<Vec<Reflex>>,
}

/// A reflex chosen to fire now.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ReflexFire {
    pub id: String,
    pub description: String,
    /// The resolved visible target, for reflexes that need one.
    pub target: Option<Landmark>,
}

/// What the nervous system decided.
#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    /// `reflex` (act now), `brain` (full LLM turn) or `ignore`.
    pub layer: String,
    pub reflex: Option<ReflexFire>,
    /// After a reflex: should the brain still produce a spoken line?
    pub reply: bool,
    /// The character declined the command.
    pub refused: bool,
    /// Probability behind the decision (reflex prob, or route prob).
    pub confidence: f32,
    /// `kev` or `rules`.
    pub source: String,
    pub latency_ms: u64,
    /// Why Kev was not used, when it fell back to the rules matcher.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
    /// Raw Kev answers (for the trace / debugging).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kev: Option<Value>,
}

impl Decision {
    fn brain(source: &str) -> Self {
        Self {
            layer: "brain".to_string(),
            reflex: None,
            reply: true,
            refused: false,
            confidence: 0.0,
            source: source.to_string(),
            latency_ms: 0,
            fallback_reason: None,
            kev: None,
        }
    }

    /// The depth-1 note handed to the brain when it runs after this decision:
    /// what the body already did, or that the character refused. `None` when
    /// there is nothing to add.
    pub fn brain_note(&self) -> Option<String> {
        if self.refused {
            let what = self
                .kev
                .as_ref()
                .and_then(|kev| kev.get("refusedReflex"))
                .and_then(Value::as_str)
                .unwrap_or("what you were told");
            return Some(format!(
                "[Instinct: you were just told to {what}, and you are NOT doing it. Don't act on it - say so in character.]"
            ));
        }
        let fired = self.reflex.as_ref()?;
        let target = fired
            .target
            .as_ref()
            .map(|t| format!(" ({})", t.label.trim()))
            .unwrap_or_default();
        Some(format!(
            "[Instinct: your body already reacted - you did this: {}{target}. Do not repeat the action; at most say a few words about it.]",
            fired.description.to_lowercase()
        ))
    }
}

/// Decides how `stimulus` is handled. Never fails: a broken Kev endpoint falls
/// back to the rules matcher, and a disabled nervous system means `brain`.
pub async fn decide(state: &AppState, settings: &AppSettings, stimulus: &Stimulus) -> Decision {
    let started = Instant::now();
    let ns = &settings.nervous_system;
    if !ns.enabled || stimulus.text.trim().is_empty() {
        return Decision::brain("disabled");
    }
    let reflexes = stimulus
        .reflexes
        .clone()
        .filter(|list| !list.is_empty())
        .unwrap_or_else(|| profile_reflexes(state, settings));
    let personality = character_summary(state, &stimulus.npc_name);
    let heard = crate::senses::recent_heard_text(&settings.senses, &stimulus.npc_name);

    let mut decision = if chasm_core::normalize_nervous_provider(&ns.provider)
        == chasm_core::NERVOUS_PROVIDER_KEV
    {
        match decide_kev(ns, stimulus, &reflexes, &personality, &heard).await {
            Ok(decision) => decision,
            Err(error) => {
                tracing::warn!("nervous system: Kev unavailable, using rules: {error}");
                let mut fallback = decide_rules(stimulus, &reflexes);
                fallback.fallback_reason = Some(error);
                fallback
            }
        }
    } else {
        decide_rules(stimulus, &reflexes)
    };
    decision.latency_ms = started.elapsed().as_millis() as u64;
    tracing::info!(
        "nervous system: {} '{}' -> {} {:?} (refused={}, reply={}, {} in {}ms)",
        stimulus.npc_name,
        truncate(&stimulus.text, 60),
        decision.layer,
        decision.reflex.as_ref().map(|r| r.id.as_str()),
        decision.refused,
        decision.reply,
        decision.source,
        decision.latency_ms,
    );
    decision
}

/// The character card's personality, trimmed — the "who is this" part of the
/// state the decision model reads (so a contrarian refuses more).
fn character_summary(state: &AppState, name: &str) -> String {
    let Ok(Some(card)) = state.repository.read_character_card(name.trim()) else {
        return String::new();
    };
    let text = if card.personality.trim().is_empty() {
        card.description
    } else {
        card.personality
    };
    truncate(text.trim(), 600)
}

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

// ---------------------------------------------------------------------------
// Kev (System One)
// ---------------------------------------------------------------------------

fn kev_client() -> &'static reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT.get_or_init(reqwest::Client::new)
}

/// Option name used for "no reflex" / "no target" in Kev choice questions.
const NONE_OPTION: &str = "none of these";

/// Unique, human-readable option names for the visible landmarks (the model
/// reads option names, so they must be meaningful), mapped back by index.
fn landmark_options(landmarks: &[Landmark]) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    for landmark in landmarks {
        let base = landmark.describe();
        let mut name = base.clone();
        let mut n = 2;
        while names.contains(&name) {
            name = format!("{base} #{n}");
            n += 1;
        }
        names.push(name);
    }
    names
}

/// Calls `POST {kev_url}/v1/systemone`. Errors are strings (they end up in the
/// decision's `fallback_reason`).
async fn system_one(ns: &NervousSystemSettings, body: &Value) -> Result<Value, String> {
    let url = format!("{}/v1/systemone", ns.kev_url.trim().trim_end_matches('/'));
    let mut request = kev_client()
        .post(&url)
        .timeout(Duration::from_millis(ns.timeout_ms.clamp(100, 30_000)))
        .json(body);
    if !ns.kev_api_key.trim().is_empty() {
        request = request.bearer_auth(ns.kev_api_key.trim());
    }
    let response = request
        .send()
        .await
        .map_err(|error| format!("Kev request failed: {error}"))?;
    let status = response.status();
    let text = response.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(format!("Kev status {status}: {}", truncate(&text, 200)));
    }
    serde_json::from_str(&text).map_err(|error| format!("Kev returned bad JSON: {error}"))
}

fn choice_answer<'a>(answers: &'a Value, id: &str) -> Option<(&'a str, f32)> {
    let answer = answers.get(id)?;
    let choice = answer.get("choice")?.as_str()?;
    let prob = answer
        .pointer(&format!(
            "/probabilities/{}",
            choice.replace('~', "~0").replace('/', "~1")
        ))
        .and_then(Value::as_f64)
        .unwrap_or(0.0) as f32;
    Some((choice, prob))
}

fn noul_answer(answers: &Value, id: &str) -> Option<f32> {
    answers.get(id)?.get("noul")?.as_f64().map(|p| p as f32)
}

async fn decide_kev(
    ns: &NervousSystemSettings,
    stimulus: &Stimulus,
    reflexes: &[Reflex],
    personality: &str,
    heard: &[String],
) -> Result<Decision, String> {
    let npc = if stimulus.npc_name.trim().is_empty() {
        "the character"
    } else {
        stimulus.npc_name.trim()
    };
    let speaker = if stimulus.speaker.trim().is_empty() {
        "The player"
    } else {
        stimulus.speaker.trim()
    };
    let landmark_names = landmark_options(&stimulus.landmarks);

    let mut state = Map::new();
    state.insert("character".into(), json!(npc));
    if !personality.is_empty() {
        state.insert("personality".into(), json!(personality));
    }
    if !stimulus.situation.trim().is_empty() {
        state.insert("situation".into(), json!(stimulus.situation.trim()));
    }
    if !stimulus.nearby_npcs.is_empty() {
        state.insert("others present".into(), json!(stimulus.nearby_npcs));
    }
    if !landmark_names.is_empty() {
        state.insert("visible".into(), json!(landmark_names));
    }
    if !heard.is_empty() {
        state.insert("recently heard".into(), json!(heard));
    }
    let event = match stimulus.kind.as_str() {
        "heard" => format!(
            "{npc} overhears {speaker} say: \"{}\"",
            stimulus.text.trim()
        ),
        "event" => format!("This just happened near {npc}: {}", stimulus.text.trim()),
        _ => format!("{speaker} says to {npc}: \"{}\"", stimulus.text.trim()),
    };
    state.insert("what just happened".into(), json!(event));

    let mut reflex_criteria = Map::new();
    for r in reflexes {
        reflex_criteria.insert(r.id.clone(), json!(r.description));
    }
    reflex_criteria.insert(NONE_OPTION.into(), json!("No single listed action fits"));

    let mut questions = json!({
        "route": {
            "type": "choice",
            "instructions": format!("How should {npc} handle what just happened?"),
            "criteria": {
                "act": format!("It is a direct instruction or a cue for {npc} to do one simple physical action right now; doing it is the natural response"),
                "think": format!("It needs {npc} to answer, explain, discuss, decide something complex, or it is a question rather than an instruction"),
                "ignore": format!("It is not meant for {npc} and needs no reaction from them"),
            },
        },
        "reflex": {
            "type": "choice",
            "instructions": format!("Which single action is {npc} being told or cued to do right now?"),
            "criteria": Value::Object(reflex_criteria),
        },
        "comply": {
            "type": "noul",
            "instructions": format!("Given their personality and the situation, would {npc} actually go along with it right now?"),
        },
        "reply": {
            "type": "noul",
            "instructions": format!("Besides acting, would {npc} naturally say something out loud in response?"),
        },
    });
    if !landmark_names.is_empty() {
        let mut criteria = Map::new();
        for name in &landmark_names {
            criteria.insert(name.clone(), Value::Null);
        }
        criteria.insert(NONE_OPTION.into(), json!("No visible thing is referred to"));
        questions["target"] = json!({
            "type": "choice",
            "instructions": "Which visible thing or spot is being referred to?",
            "criteria": Value::Object(criteria),
        });
    }
    let body = json!({
        "state": Value::Object(state),
        "model": ns.kev_model,
        "questions": questions,
    });
    let response = system_one(ns, &body).await?;
    let answers = response
        .get("answers")
        .ok_or_else(|| "Kev response has no answers".to_string())?;

    let (route, route_p) = choice_answer(answers, "route").ok_or("Kev: no route answer")?;
    let (reflex_id, reflex_p) = choice_answer(answers, "reflex").unwrap_or((NONE_OPTION, 0.0));
    let comply = noul_answer(answers, "comply").unwrap_or(1.0);
    let reply = noul_answer(answers, "reply").unwrap_or(0.5) >= 0.5;
    let target = choice_answer(answers, "target")
        .filter(|(choice, _)| *choice != NONE_OPTION)
        .and_then(|(choice, _)| landmark_names.iter().position(|n| n == choice))
        .map(|index| stimulus.landmarks[index].clone());

    let mut decision = Decision::brain("kev");
    decision.confidence = route_p;
    decision.kev = Some(answers.clone());
    match route {
        "ignore" if stimulus.kind != "speech" || !names_npc(&stimulus.text, npc) => {
            decision.layer = "ignore".to_string();
            decision.reply = false;
        }
        "act" => {
            let chosen = reflexes.iter().find(|r| r.id == reflex_id);
            if let Some(chosen) = chosen.filter(|_| reflex_p >= ns.reflex_threshold) {
                if comply < ns.refuse_threshold {
                    decision.refused = true;
                    decision.confidence = 1.0 - comply;
                    if let Some(kev) = decision.kev.as_mut() {
                        kev["refusedReflex"] = json!(chosen.description.to_lowercase());
                    }
                } else if chosen.needs_target && target.is_none() {
                    // Told to go somewhere we can't place: let the brain ask.
                } else {
                    decision.layer = "reflex".to_string();
                    decision.confidence = reflex_p;
                    decision.reply = reply;
                    decision.reflex = Some(ReflexFire {
                        id: chosen.id.clone(),
                        description: chosen.description.clone(),
                        target: if chosen.needs_target { target } else { None },
                    });
                }
            }
        }
        _ => {}
    }
    Ok(decision)
}

// ---------------------------------------------------------------------------
// Rules matcher
// ---------------------------------------------------------------------------

/// Leading words that turn a sentence into a question (→ the brain).
const QUESTION_STARTS: &[&str] = &[
    "can", "could", "would", "will", "do", "does", "did", "are", "is", "was", "what", "why", "how",
    "where", "who", "when", "should", "have", "has", "which",
];
/// Lead-ins stripped before matching an imperative ("okay, I want you to …").
const LEAD_INS: &[&str] = &[
    "i want you to",
    "i need you to",
    "you need to",
    "i want you",
    "go ahead and",
    "alright",
    "all right",
    "okay",
    "ok",
    "right",
    "please",
    "hey",
    "yo",
    "now",
    "quick",
    "quickly",
    "just",
    "and",
    "then",
    "so",
];
/// Words that may trail a one-word command without changing it ("jump now").
const FILLER: &[&str] = &[
    "now", "please", "quick", "quickly", "already", "again", "mate", "bro", "man", "buddy", "dude",
    "then", "right", "ok", "okay", "for", "me", "go",
];

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .replace(['’', '\''], "")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

fn names_npc(text: &str, npc: &str) -> bool {
    let text_words = words(text);
    let name_words = words(npc);
    !name_words.is_empty() && name_words.iter().all(|w| text_words.contains(w))
}

fn starts_with_words(tokens: &[String], phrase: &[String]) -> bool {
    phrase.len() <= tokens.len() && tokens.iter().zip(phrase).all(|(a, b)| a == b)
}

/// Removes vocatives (the NPC's name, or any present character's name) and
/// lead-in words from the front, and a trailing vocative from the back.
fn strip_address(mut tokens: Vec<String>, names: &[&str]) -> Vec<String> {
    let names: Vec<Vec<String>> = names
        .iter()
        .map(|n| words(n))
        .filter(|n| !n.is_empty())
        .collect();
    loop {
        let before = tokens.len();
        for name in &names {
            if starts_with_words(&tokens, name) {
                tokens.drain(..name.len());
            }
            if tokens.len() >= name.len() && tokens[tokens.len() - name.len()..] == name[..] {
                tokens.truncate(tokens.len() - name.len());
            }
        }
        for lead in LEAD_INS {
            let lead_words = words(lead);
            if starts_with_words(&tokens, &lead_words) {
                tokens.drain(..lead_words.len());
                break;
            }
        }
        if tokens.len() == before {
            return tokens;
        }
    }
}

/// Stems a word for loose matching ("bodies" → "bodi", "rocks" → "rock").
fn stem(word: &str) -> String {
    let w = word.trim_end_matches("es").trim_end_matches('s');
    let w = w.trim_end_matches('y').trim_end_matches('i');
    if w.len() >= 3 {
        w.to_string()
    } else {
        word.to_string()
    }
}

/// Picks the landmark whose label best overlaps `tokens` (ties → nearest).
fn match_landmark(tokens: &[String], landmarks: &[Landmark]) -> Option<(Landmark, f32)> {
    let wanted: Vec<String> = tokens.iter().map(|t| stem(t)).collect();
    let mut best: Option<(usize, f64, &Landmark)> = None;
    for landmark in landmarks {
        let label: Vec<String> = words(&format!("{} {}", landmark.label, landmark.kind))
            .iter()
            .map(|w| stem(w))
            .collect();
        let overlap = label
            .iter()
            .filter(|w| w.len() > 2 && wanted.contains(w))
            .count();
        if overlap == 0 {
            continue;
        }
        let distance = landmark.distance_m.unwrap_or(f64::MAX);
        let better = match best {
            None => true,
            Some((o, d, _)) => overlap > o || (overlap == o && distance < d),
        };
        if better {
            best = Some((overlap, distance, landmark));
        }
    }
    best.map(|(overlap, _, landmark)| {
        let confidence = (0.6 + 0.15 * overlap as f32).min(0.95);
        (landmark.clone(), confidence)
    })
}

/// Best landmark for a free-text name/reference by word overlap.
pub fn match_landmark_text(text: &str, landmarks: &[Landmark]) -> Option<Landmark> {
    match_landmark(&words(text), landmarks).map(|(landmark, _)| landmark)
}

/// The rules matcher: imperative phrases at the start of a clause fire their
/// reflex; questions and anything unmatched go to the brain.
fn decide_rules(stimulus: &Stimulus, reflexes: &[Reflex]) -> Decision {
    let mut decision = Decision::brain("rules");
    let npc = stimulus.npc_name.trim();
    // Talking to someone else in the group (named, and not us): stay out of it.
    if stimulus.kind == "speech"
        && !names_npc(&stimulus.text, npc)
        && stimulus
            .nearby_npcs
            .iter()
            .any(|other| !other.eq_ignore_ascii_case(npc) && names_npc(&stimulus.text, other))
    {
        decision.layer = "ignore".to_string();
        decision.reply = false;
        decision.confidence = 0.7;
        return decision;
    }
    // Overheard lines and events never trigger a commanded reflex in the rules
    // matcher (that needs judgment — Kev); they just inform the brain.
    if stimulus.kind == "heard" || stimulus.kind == "event" {
        return decision;
    }
    let text = stimulus.text.trim();
    // Judge the first clause: "Kevin, jump. There you go" → "Kevin, jump".
    let clause = text
        .split(['.', '!', ';'])
        .map(str::trim)
        .find(|c| !words(c).is_empty())
        .unwrap_or(text);
    let mut names: Vec<&str> = vec![npc];
    names.extend(stimulus.nearby_npcs.iter().map(String::as_str));
    let tokens = strip_address(words(clause), &names);
    if tokens.is_empty() {
        return decision;
    }
    let is_question =
        clause.trim_end().ends_with('?') || QUESTION_STARTS.contains(&tokens[0].as_str());
    if is_question {
        return decision;
    }

    // Longest matching phrase wins ("get in the car" over "get in").
    let mut best: Option<(&Reflex, usize)> = None;
    for r in reflexes {
        for phrase in &r.phrases {
            let phrase_words = words(phrase);
            if phrase_words.is_empty() || !starts_with_words(&tokens, &phrase_words) {
                continue;
            }
            let rest = &tokens[phrase_words.len()..];
            let fits = phrase_words.len() > 1
                || r.needs_target
                || rest.iter().all(|w| FILLER.contains(&w.as_str()));
            if fits && best.map_or(true, |(_, len)| phrase_words.len() > len) {
                best = Some((r, phrase_words.len()));
            }
        }
    }
    let Some((chosen, phrase_len)) = best else {
        return decision;
    };
    let rest = &tokens[phrase_len..];
    let mut target = None;
    let mut confidence = 0.9;
    if chosen.needs_target {
        match match_landmark(rest, &stimulus.landmarks) {
            Some((landmark, c)) => {
                target = Some(landmark);
                confidence = c;
            }
            // Can't tell where: the brain can ask or pick from what it sees.
            None => return decision,
        }
    }
    let total_words = words(text).len();
    let command_words =
        words(npc).len() + phrase_len + if target.is_some() { rest.len() } else { 0 };
    decision.layer = "reflex".to_string();
    decision.confidence = confidence;
    // More than the bare command was said: the brain gets a word in too.
    decision.reply = total_words > command_words + 2 || text.contains('?');
    decision.reflex = Some(ReflexFire {
        id: chosen.id.clone(),
        description: chosen.description.clone(),
        target,
    });
    decision
}

/// A spoken reference resolved to a visible landmark.
pub struct Grounding {
    pub target: Option<Landmark>,
    pub confidence: f32,
    pub source: String,
}

/// Resolves "those bodies over there" to one of `landmarks`, with Kev when it
/// is the configured provider, else (or on failure) by label overlap.
pub async fn ground_reference(
    settings: &AppSettings,
    npc_name: &str,
    text: &str,
    landmarks: &[Landmark],
) -> Grounding {
    let ns = &settings.nervous_system;
    if ns.enabled
        && chasm_core::normalize_nervous_provider(&ns.provider) == chasm_core::NERVOUS_PROVIDER_KEV
    {
        let names = landmark_options(landmarks);
        let mut criteria = Map::new();
        for name in &names {
            criteria.insert(name.clone(), Value::Null);
        }
        criteria.insert(NONE_OPTION.into(), json!("No visible thing is referred to"));
        let body = json!({
            "state": {
                "character": npc_name,
                "visible": names,
                "what was said": text,
            },
            "model": ns.kev_model,
            "questions": { "target": {
                "type": "choice",
                "instructions": "Which visible thing or spot is being referred to?",
                "criteria": Value::Object(criteria),
            }},
        });
        match system_one(ns, &body).await {
            Ok(response) => {
                let answer = response
                    .get("answers")
                    .and_then(|a| choice_answer(a, "target"));
                if let Some((choice, prob)) = answer {
                    let target = names
                        .iter()
                        .position(|n| n == choice)
                        .map(|index| landmarks[index].clone());
                    return Grounding {
                        target,
                        confidence: prob,
                        source: "kev".to_string(),
                    };
                }
            }
            Err(error) => tracing::warn!("look: Kev unavailable, using rules: {error}"),
        }
    }
    let (target, confidence) = match match_landmark(&words(text), landmarks) {
        Some((landmark, confidence)) => (Some(landmark), confidence),
        None => (None, 0.0),
    };
    Grounding {
        target,
        confidence,
        source: "rules".to_string(),
    }
}

/// `POST /api/game/v1/stimulus`: run the nervous system on one stimulus and
/// return its decision (no LLM involved). Bridges call this on every player
/// utterance / notable event and fire `decision.reflex` immediately; when
/// `layer` is `brain` (or `reply` is set) they follow up with a normal turn.
pub async fn stimulus(
    State(state): State<Arc<AppState>>,
    Json(req): Json<Stimulus>,
) -> WebResult<Json<Decision>> {
    let settings = AppSettings::load(&state.config.settings_path);
    Ok(Json(decide(&state, &settings, &req).await))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn speech(text: &str) -> Stimulus {
        Stimulus {
            npc_name: "Kevin".into(),
            kind: "speech".into(),
            speaker: "player".into(),
            text: text.into(),
            ..Default::default()
        }
    }

    fn rules(stimulus: &Stimulus) -> Decision {
        decide_rules(stimulus, &default_reflexes())
    }

    fn fired(decision: &Decision) -> Option<&str> {
        decision.reflex.as_ref().map(|r| r.id.as_str())
    }

    #[test]
    fn bare_commands_fire_reflexes() {
        for (text, id) in [
            ("Kevin, jump.", "jump"),
            ("Kevin jump", "jump"),
            ("jump now kevin", "jump"),
            ("Kevin, crouch.", "crouch"),
            ("Kevin, stand up.", "stand"),
            (
                "Okay, I want you to hold your fire until I say so.",
                "hold_fire",
            ),
            ("Get in the car, Kevin.", "enter_vehicle"),
            (
                "Kevin, I want you to wait here rather than going in.",
                "wait",
            ),
            ("Shoot!", "fire_at_will"),
        ] {
            let decision = rules(&speech(text));
            assert_eq!(decision.layer, "reflex", "{text}");
            assert_eq!(fired(&decision), Some(id), "{text}");
        }
    }

    #[test]
    fn questions_and_chat_go_to_the_brain() {
        for text in [
            "Kevin, can you jump?",
            "What are we supposed to be doing right now?",
            "Kevin, what's your opinion on grunts?",
            "Jumping is overrated, isn't it",
        ] {
            let decision = rules(&speech(text));
            assert_eq!(decision.layer, "brain", "{text}");
            assert!(decision.reflex.is_none(), "{text}");
        }
    }

    #[test]
    fn one_word_commands_need_a_bare_clause() {
        // "fire" followed by real content is not a bare command.
        assert_eq!(
            rules(&speech("Fire the rockets at the tower over there")).layer,
            "brain"
        );
        assert_eq!(rules(&speech("Wait, what was that noise?")).layer, "brain");
    }

    #[test]
    fn extra_talk_asks_the_brain_to_reply_too() {
        let bare = rules(&speech("Kevin, jump."));
        assert!(!bare.reply);
        let chatty = rules(&speech(
            "Kevin, jump. Come on, show them what a Spartan can do",
        ));
        assert_eq!(fired(&chatty), Some("jump"));
        assert!(chatty.reply);
    }

    #[test]
    fn move_to_resolves_a_visible_landmark() {
        let mut stimulus = speech("Hey Kevin, can you go stand near those bodies over there");
        stimulus.landmarks = vec![
            Landmark {
                id: "rock".into(),
                label: "large rock".into(),
                distance_m: Some(6.0),
                ..Default::default()
            },
            Landmark {
                id: "bodies".into(),
                label: "pile of Marine bodies".into(),
                distance_m: Some(20.0),
                ..Default::default()
            },
        ];
        // A polite question is still a question → brain (Kev would act on it).
        assert_eq!(rules(&stimulus).layer, "brain");
        stimulus.text = "Kevin, go stand near those bodies over there".into();
        let decision = rules(&stimulus);
        assert_eq!(fired(&decision), Some("move_to"));
        assert_eq!(decision.reflex.unwrap().target.unwrap().id, "bodies");
        stimulus.text = "Stand by this rock".into();
        assert_eq!(rules(&stimulus).reflex.unwrap().target.unwrap().id, "rock");
        // No matching landmark → the brain sorts it out.
        stimulus.text = "Go stand by the tower".into();
        assert_eq!(rules(&stimulus).layer, "brain");
    }

    #[test]
    fn talking_to_someone_else_is_ignored() {
        let mut stimulus = speech("Johnson, get in the car");
        stimulus.nearby_npcs = vec!["Kevin".into(), "Johnson".into()];
        assert_eq!(rules(&stimulus).layer, "ignore");
        stimulus.text = "Johnson and Kevin, get in the car".into();
        assert_eq!(fired(&rules(&stimulus)), Some("enter_vehicle"));
    }

    #[test]
    fn brain_note_describes_the_reflex_or_refusal() {
        let decision = rules(&speech("Kevin, jump"));
        assert!(decision.brain_note().unwrap().contains("jump in place"));
        let mut refused = Decision::brain("kev");
        refused.refused = true;
        refused.kev = Some(json!({ "refusedReflex": "jump in place" }));
        assert!(refused.brain_note().unwrap().contains("NOT doing it"));
    }

    /// A tiny canned System One server: answers `route`/`reflex`/… from a fixed
    /// table so the Kev client path (request shape, answer parsing, thresholds)
    /// is exercised end to end over real HTTP.
    async fn mock_kev(
        answers: Value,
    ) -> (
        String,
        tokio::task::JoinHandle<()>,
        Arc<std::sync::Mutex<Option<Value>>>,
    ) {
        use axum::routing::post;
        let seen: Arc<std::sync::Mutex<Option<Value>>> = Arc::default();
        let seen_handler = seen.clone();
        let app = axum::Router::new().route(
            "/v1/systemone",
            post(move |Json(body): Json<Value>| {
                let answers = answers.clone();
                let seen = seen_handler.clone();
                async move {
                    *seen.lock().unwrap() = Some(body);
                    Json(json!({ "model": "kev-latest", "answers": answers, "latency_ms": 7 }))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}"), handle, seen)
    }

    fn kev_settings(url: &str) -> NervousSystemSettings {
        NervousSystemSettings {
            provider: "kev".into(),
            kev_url: url.into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn kev_fires_a_targeted_reflex() {
        let (url, server, seen) = mock_kev(json!({
            "route": { "type": "choice", "choice": "act", "probabilities": { "act": 0.91, "think": 0.06, "ignore": 0.03 } },
            "reflex": { "type": "choice", "choice": "move_to", "probabilities": { "move_to": 0.88 } },
            "comply": { "type": "noul", "noul": 0.8 },
            "reply": { "type": "noul", "noul": 0.7 },
            "target": { "type": "choice", "choice": "pile of bodies (20m)", "probabilities": { "pile of bodies (20m)": 0.9 } },
        }))
        .await;
        let mut stimulus = speech("Hey Kevin, can you go stand near those bodies?");
        stimulus.landmarks = vec![
            Landmark {
                id: "rock".into(),
                label: "rock".into(),
                distance_m: Some(5.0),
                ..Default::default()
            },
            Landmark {
                id: "bodies".into(),
                label: "pile of bodies".into(),
                distance_m: Some(20.0),
                ..Default::default()
            },
        ];
        let decision = decide_kev(
            &kev_settings(&url),
            &stimulus,
            &default_reflexes(),
            "Abrasive.",
            &[],
        )
        .await
        .expect("kev decision");
        server.abort();
        assert_eq!(decision.layer, "reflex");
        assert_eq!(decision.source, "kev");
        assert!(decision.reply);
        let fire = decision.reflex.unwrap();
        assert_eq!(fire.id, "move_to");
        assert_eq!(fire.target.unwrap().id, "bodies");
        // The request carried the state + all question types.
        let body = seen.lock().unwrap().clone().unwrap();
        assert_eq!(body["model"], "kev-latest");
        assert!(body["state"]["what just happened"]
            .as_str()
            .unwrap()
            .contains("stand near those bodies"));
        assert_eq!(body["questions"]["comply"]["type"], "noul");
        assert!(body["questions"]["target"]["criteria"]
            .get("rock (5m)")
            .is_some());
        assert!(body["questions"]["reflex"]["criteria"]
            .get(NONE_OPTION)
            .is_some());
    }

    #[tokio::test]
    async fn kev_refusal_goes_to_the_brain_with_a_note() {
        let (url, server, _) = mock_kev(json!({
            "route": { "type": "choice", "choice": "act", "probabilities": { "act": 0.8 } },
            "reflex": { "type": "choice", "choice": "jump", "probabilities": { "jump": 0.95 } },
            "comply": { "type": "noul", "noul": 0.05 },
            "reply": { "type": "noul", "noul": 0.9 },
        }))
        .await;
        let decision = decide_kev(
            &kev_settings(&url),
            &speech("Kevin, jump"),
            &default_reflexes(),
            "",
            &[],
        )
        .await
        .unwrap();
        server.abort();
        assert_eq!(decision.layer, "brain");
        assert!(decision.refused);
        assert!(decision.reflex.is_none());
        assert!(decision.brain_note().unwrap().contains("jump in place"));
    }

    #[tokio::test]
    async fn kev_low_confidence_reflex_escalates() {
        let (url, server, _) = mock_kev(json!({
            "route": { "type": "choice", "choice": "act", "probabilities": { "act": 0.5 } },
            "reflex": { "type": "choice", "choice": "jump", "probabilities": { "jump": 0.3 } },
            "comply": { "type": "noul", "noul": 0.9 },
            "reply": { "type": "noul", "noul": 0.1 },
        }))
        .await;
        let decision = decide_kev(
            &kev_settings(&url),
            &speech("Kevin, jump"),
            &default_reflexes(),
            "",
            &[],
        )
        .await
        .unwrap();
        server.abort();
        assert_eq!(decision.layer, "brain");
        assert!(!decision.refused);
    }

    #[tokio::test]
    async fn kev_unreachable_is_an_error_for_fallback() {
        // Port 9 (discard) on localhost: connection refused.
        let error = decide_kev(
            &kev_settings("http://127.0.0.1:9"),
            &speech("jump"),
            &default_reflexes(),
            "",
            &[],
        )
        .await
        .unwrap_err();
        assert!(error.contains("Kev request failed"), "{error}");
    }
}
