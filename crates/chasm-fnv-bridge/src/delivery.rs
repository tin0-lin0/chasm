//! Voice direction: delivery cues and vocal events in generated lines.
//!
//! With voice direction on, the brain may open a line with a bracketed delivery
//! cue and sprinkle inline vocal events:
//!
//! ```text
//! [whispering] Shut your damn trap. (sigh) Grunt at two o'clock.
//! ```
//!
//! [`split`] separates the three audiences of that text:
//!
//! * the **cue** (`whispering`) — forwarded to the TTS engine as voice-direction
//!   `instructions` (Breeze TTS 2) and to the bridge (e.g. to lower the volume);
//! * the **speech** text (`Shut your damn trap. (sigh) Grunt at two o'clock.`) —
//!   what the engine synthesizes, vocal events kept so it can perform them;
//! * the **caption** (`Shut your damn trap. Grunt at two o'clock.`) — what the
//!   player reads, with every cue and vocal event removed.
//!
//! The brain streams a line sentence by sentence, so a cue at the start of a
//! line carries over to the following sentences until a new cue replaces it
//! ([`CueTracker`]).

use std::sync::OnceLock;

use regex::Regex;

/// Inline non-verbal sounds the TTS engine can perform. Only these are treated
/// as vocal events — any other parenthetical is ordinary text and stays.
pub const VOCAL_EVENTS: &[&str] = &[
    "laugh",
    "laughs",
    "laughing",
    "chuckle",
    "chuckles",
    "giggle",
    "sigh",
    "sighs",
    "cough",
    "coughs",
    "clears throat",
    "gasp",
    "gasps",
    "groan",
    "groans",
    "grunt",
    "grunts",
    "sniff",
    "sniffs",
    "scoff",
    "scoffs",
    "snort",
    "snorts",
    "scream",
    "screams",
    "yell",
    "shout",
    "sob",
    "whistle",
    "whistles",
    "hum",
    "hums",
    "growl",
    "growls",
    "pant",
    "panting",
    "breath",
    "breathes",
    "inhales",
    "exhales",
    "spits",
];

fn leading_cue_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*\[([^\[\]]{1,80})\]\s*").expect("leading cue regex"))
}

fn any_cue_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s*\[[^\[\]]{1,80}\]\s*").expect("cue regex"))
}

fn paren_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s*\(([^()]{1,40})\)\s*").expect("paren regex"))
}

fn space_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s{2,}").expect("space regex"))
}

fn space_before_punct_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\s+([,.!?;:…])").expect("punct regex"))
}

fn is_vocal_event(inner: &str) -> bool {
    let inner = inner.trim().to_lowercase();
    VOCAL_EVENTS.contains(&inner.as_str())
}

fn tidy(text: &str) -> String {
    let collapsed = space_re().replace_all(text, " ");
    space_before_punct_re()
        .replace_all(&collapsed, "$1")
        .trim()
        .to_string()
}

/// One line split for its three audiences (see the module docs).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Delivery {
    /// The leading delivery cue, without brackets (`whispering`).
    pub cue: Option<String>,
    /// Text for the TTS engine: cues removed, vocal events kept.
    pub speech: String,
    /// Text for subtitles: cues and vocal events removed.
    pub caption: String,
}

/// Splits `text` into cue / speech / caption. `keep_vocal_events = false`
/// strips vocal events from the speech too (for engines that would read
/// "(laugh)" aloud).
pub fn split(text: &str, keep_vocal_events: bool) -> Delivery {
    let cue = leading_cue_re()
        .captures(text)
        .map(|c| c[1].trim().to_string())
        .filter(|c| !c.is_empty());
    let without_cues = any_cue_re().replace_all(text, " ");
    let strip_events = |input: &str| {
        paren_re()
            .replace_all(input, |caps: &regex::Captures| {
                if is_vocal_event(&caps[1]) {
                    " ".to_string()
                } else {
                    caps[0].to_string()
                }
            })
            .into_owned()
    };
    let caption = tidy(&strip_events(&without_cues));
    let speech = if keep_vocal_events {
        tidy(&without_cues)
    } else {
        caption.clone()
    };
    Delivery {
        cue,
        speech,
        caption,
    }
}

/// Caption-only cleanup (cues and vocal events removed).
pub fn caption(text: &str) -> String {
    split(text, false).caption
}

/// Carries a line's delivery cue across its streamed sentences.
#[derive(Debug, Default)]
pub struct CueTracker {
    current: Option<String>,
}

impl CueTracker {
    /// Splits one streamed segment; a segment without its own leading cue
    /// inherits the last one seen.
    pub fn next(&mut self, segment: &str, keep_vocal_events: bool) -> Delivery {
        let mut delivery = split(segment, keep_vocal_events);
        match &delivery.cue {
            Some(cue) => self.current = Some(cue.clone()),
            None => delivery.cue = self.current.clone(),
        }
        delivery
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_cue_events_and_caption() {
        let d = split(
            "[whispering] Shut your damn trap. (sigh) Grunt at two o'clock.",
            true,
        );
        assert_eq!(d.cue.as_deref(), Some("whispering"));
        assert_eq!(
            d.speech,
            "Shut your damn trap. (sigh) Grunt at two o'clock."
        );
        assert_eq!(d.caption, "Shut your damn trap. Grunt at two o'clock.");
        let no_events = split("[whispering] Quiet (laugh) now.", false);
        assert_eq!(no_events.speech, "Quiet now.");
    }

    #[test]
    fn ordinary_parentheticals_stay() {
        let d = split("Take the Warthog (the one with the gun) and go.", true);
        assert_eq!(d.cue, None);
        assert_eq!(d.caption, "Take the Warthog (the one with the gun) and go.");
    }

    #[test]
    fn inline_brackets_never_reach_captions_or_speech() {
        let d = split("Meow. [purring] Damn it, Chasm.", true);
        assert_eq!(d.cue, None);
        assert_eq!(d.caption, "Meow. Damn it, Chasm.");
        assert_eq!(d.speech, "Meow. Damn it, Chasm.");
    }

    #[test]
    fn cue_carries_across_sentences() {
        let mut tracker = CueTracker::default();
        assert_eq!(
            tracker
                .next("[thick German accent] Listen, you fool.", true)
                .cue
                .as_deref(),
            Some("thick German accent")
        );
        assert_eq!(
            tracker
                .next("He is the ultimate soldier.", true)
                .cue
                .as_deref(),
            Some("thick German accent")
        );
        assert_eq!(
            tracker.next("[shouting] Die!", true).cue.as_deref(),
            Some("shouting")
        );
    }
}
