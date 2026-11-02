# Halo CE squadmate: nervous system, senses and voice

> **Audience:** whoever writes the game side for Halo: Combat Evolved (the
> [OpenCE](https://github.com/OpenCommunityEdition/OpenCE) decompilation) or any
> other game that wants a "Kevin": an AI squadmate who reacts instantly, sees and
> hears the battlefield, and talks with a directed voice.
>
> Everything here is **additive** to the contract in
> [building-a-bridge-mod.md](building-a-bridge-mod.md). It is game-agnostic; Halo
> is the worked example. The `profiles/halo-ce/` profile and
> `scripts/halo_sim.py` (a fake Halo bridge) exercise all of it.

---

## 1. The two layers

```
 player speaks ──► STT ──► NERVOUS SYSTEM ──► reflex (instant, no LLM) ──► bridge executes
                               │
                               └─ needs thought/speech ──► BRAIN (LLM turn) ──► speech + actions
```

- **Nervous system** (`nervous_system` settings): one fast, non-generative pass
  over every stimulus. It decides whether this is a plain physical command
  ("Kevin, jump"), a question ("can you jump?"), something to ignore (talking
  to someone else), or a command the character **refuses**. Reflexes stream to
  the bridge before any LLM work starts.
  - `provider: "kev"` uses a [Kev](https://github.com/jaredpalmer/kev)
    decision model (e.g. `jaredpalmer/kev-4b`) through its System One endpoint.
    One request asks typed questions (route / reflex / comply / reply /
    target) and gets calibrated probabilities back with no text generation.
  - `provider: "rules"` (default) is a built-in imperative matcher over the
    reflex phrases. It needs no model, and it is also the automatic fallback
    when Kev is down or slow.
- **Brain**: the normal chasm turn (the LLM, e.g. Gemma 4 12B). It now also
  sees what the NPC can see, what it overheard, the mission objective, and what
  its body already did (`[Instinct: …]`).

A reflex can still be followed by speech. "Kevin, jump. Show me what a Spartan
can do" fires `body.jump` instantly, then the brain says something. The brain
is told it already jumped, so it doesn't jump twice.

---

## 2. Endpoints

All under `http://127.0.0.1:7341`.

### `POST /api/game/v1/stimulus`: nervous system only

```jsonc
{
  "npc_name": "Kevin",
  "kind": "speech",               // speech | heard | event
  "speaker": "player",
  "text": "Kevin, go stand near those dead Marines",
  "landmarks": [ /* see §3 */ ],
  "nearby_npcs": ["Kevin", "Sgt. Johnson"],   // to tell who is addressed
  "situation": "Location: Halo. In combat.",  // optional, Kev reads it
  "reflexes": null                // optional per-request catalog override
}
```

Response (a `Decision`):

```jsonc
{
  "layer": "reflex",              // reflex | brain | ignore
  "reflex": { "id": "squad.move_to", "description": "Walk over to …",
              "target": { "id": "bodies_01", "label": "pile of dead Marines",
                          "position": { "x": 131.0, "y": -12.5, "z": 3.2 }, … } },
  "reply": false,                 // after the reflex, should the brain still talk?
  "refused": false,               // the character declined (brain voices it)
  "confidence": 0.88,
  "source": "kev",                // kev | rules
  "latency_ms": 41,
  "fallback_reason": null         // set when Kev failed and rules decided
}
```

### `POST /api/game/v1/hear`: overheard dialogue and radio

Send every in-game voice line the NPC could hear: Marines' barks, Covenant
chatter, Cortana in the player's ear. Lines are kept in a short rolling window
(`senses.hearing_window_secs`, 180 s) and shown to the brain as "voices you
recently overheard".

```jsonc
{ "speaker": "Cortana", "channel": "radio",      // world | radio
  "text": "They're trying to flank us.",
  "listeners": ["Kevin"],                         // empty = everyone in earshot
  "react": true }                                 // also run each listener's nervous system
```

`react: true` returns `reactions[].decision` per listener, so a bridge can let
a heard line trigger a reflex. "Grenade!" is a good example.

A turn may also carry lines inline as `metadata.heard: [{speaker, text, channel}]`.

### `POST /api/game/v1/look`: sight

```jsonc
{ "npc_name": "Kevin", "text": "that Warthog behind us",
  "landmarks": [ … ], "screenshot_base64": "" }
→ { "target": { …landmark… }, "confidence": 0.75, "source": "rules", "description": null }
```

With `senses.screenshot_enabled` and a vision model (Gemma 4 with its mmproj on
the local llama-server, or any OpenAI-compatible vision endpoint), a screenshot
is turned into a short scene description.

### `POST /api/game/v1/turn`: new metadata and events

New optional `metadata` keys:

| Key | Meaning |
| --- | --- |
| `vision.landmarks` | What the NPC can see (§3). Injected as "what you can see"; used to ground `move_to`. |
| `vision.screenshot_base64` | Optional view screenshot; described once, never stored. |
| `heard` | Lines overheard since the last turn. |
| `objective` | Current mission objective (e.g. from the HUD/scenario script). |
| `radio_capable` | The NPC carries a radio; beyond `voice_direction.radio_distance_m` (25 m) it talks over comms instead of being out of earshot. |
| `radio` | Force radio delivery for this turn. |
| `in_combat` | Already supported; also passed to the nervous system. |
| `nervous_skip` | Skip the nervous system for this turn. |

New NDJSON events, in order:

| Event | When |
| --- | --- |
| `nervous` | First line of every player-speech turn: the full `Decision`. |
| `reflex` | Right after, when a reflex fires: `{action, description, target, confidence, source, latencyMs}`. **Execute it immediately.** |
| `reply` + `turn.completed` | Straight after the reflex when the brain is not needed (`layer` reflex with `reply: false`, or `ignore`). |
| `audio.chunk` | As before, plus `delivery` (the voice cue, e.g. `"whispering"`) and, on radio turns, `channel: "radio"` + `nonPositional: true` (audio already radio-filtered). |
| `action` | As before, plus `target`, `parameters` and `resolvedTarget` (the landmark with coordinates when the target names something visible). |

---

## 3. Landmarks

The bridge decides what is notable in the NPC's view and reports it each turn.
On OpenCE, a cheap scan of nearby objects is enough: scenery, vehicles, bipeds
(dead or alive), device machines, weapons. Filter them by distance and line of
sight from the NPC's head.

```jsonc
{ "id": "rock_01",               // bridge-side id, echoed back in targets
  "label": "large boulder",      // plain words; this is what the player will say
  "kind": "cover",               // optional: cover | vehicle | corpse | enemy | wreck …
  "distance_m": 8.0,
  "direction": "ahead-left",     // optional coarse direction from the NPC
  "position": { "x": 112.4, "y": -36.0, "z": 4.1 } }
```

Good labels make grounding work. Use "pile of dead Marines" rather than
"marine_body_03". Merge clusters ("three dead Marines" becomes one landmark).

---

## 4. Action vocabulary (Halo CE profile)

Reflex ids and the action book's `actionId`s are the same, so the bridge
implements one switch whichever layer chose the action:

| id | Effect | OpenCE hook (suggested) |
| --- | --- | --- |
| `body.jump` | Jump | set the unit's jump control flag for one tick |
| `body.crouch` / `body.stand` | Crouch / stand | unit crouch control flag |
| `squad.follow` | Follow the player | AI follow / `ai_follow_target_players` |
| `squad.wait` | Hold position | clear follow; AI hold at current point |
| `squad.move_to` | Walk to a landmark (`target.position`) | AI command list / move-to-point |
| `combat.hold_fire` / `combat.fire_at_will` | Weapons tight / free | disable / enable the actor's firing (keep a flag the AI firing decision checks) |
| `combat.attack` | Attack `target` (anyone, including the player) | set the actor's prop/target, force hostile |
| `combat.take_cover` | Take cover | AI cover search |
| `combat.throw_grenade` | Throw a grenade | actor grenade throw |
| `combat.reload` | Reload | unit reload |
| `vehicle.enter` / `vehicle.exit` | Get in / out (`target` = seat) | `vehicle_load_magic` / `unit_enter_vehicle` / exit |
| `body.gesture` | Wave / salute / point (`target`) | play a unit animation |

The book binds every entry to the engine `halo-ce:http-bridge`. chasm treats
`<game>:http-bridge` bindings as trusted (like FNV's xNVSE scripts) and hands the
action to the bridge by id instead of building an FNV command file.

---

## 5. Voice: direction, vocal events, radio

- **Voice direction** (`voice_direction.enabled`, or `features.voiceDirection`
  in `profile.json`, which the Halo profile sets). The brain may open a line with a
  cue, as in `[whispering] Grunt at two o'clock. (sigh) Stay low.` chasm sends
  the cue to TTS as `instructions`, keeps `(sigh)` in the spoken text for
  engines that perform it, and strips both from captions. A cue carries across
  the sentences of one line.
- **Breeze TTS 2** (`tts.local_engine = "breeze-tts-2"`). It handles voice clone,
  voice direction and vocal events. Breeze runs on Linux + CUDA, so run its API
  natively, in WSL2, or from its Docker image:
  ```bash
  python -m breeze_infer.api BreezeBlue/breeze-tts-2 --host 0.0.0.0 --port 7860
  ```
  chasm launches `scripts/breeze_tts_server.py` on `:5002`, a stdlib adapter
  that forwards to `CHASM_BREEZE_URL` (default `http://127.0.0.1:7860`). Breeze
  needs each reference clip's exact transcript. The adapter uses
  `voices/<name>/reference.txt` if present, otherwise it transcribes the clip
  once through chasm's own STT and caches it. Note: Breeze TTS 2 weights are
  licensed for research and non-commercial use only.
- **Radio**. On radio turns chasm band-limits the audio (≈300 Hz to 3.2 kHz), adds
  saturation, static (`voice_direction.radio_static`) and an opening squelch,
  and tells the brain it is on comms ("keep it short").

---

## 6. Setting it up

1. Copy `profiles/halo-ce/` into chasm's profiles folder and select it (Settings
   → Profiles). Rebuild the card PNG after editing
   `source/Kevin.card.json`: `python profiles/halo-ce/source/build_cards.py`.
2. Put a voice clip for Kevin at `voices/Kevin/reference.wav`, plus
   `reference.txt` with its exact transcript for Breeze.
3. Optional Kev nervous system:
   ```bash
   git clone https://github.com/jaredpalmer/kev.git && cd kev && uv sync --extra serve
   uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8008
   ```
   then in `chasm.settings.json`:
   ```json
   "nervous_system": { "provider": "kev", "kev_url": "http://127.0.0.1:8008" }
   ```
4. Try it without the game:
   ```bash
   python scripts/halo_sim.py                # nervous system + senses
   python scripts/halo_sim.py --with-brain   # plus full LLM turns
   python scripts/halo_sim.py --interactive  # talk to Kevin
   ```

### Settings reference (`chasm.settings.json`)

```jsonc
"nervous_system": { "enabled": true, "provider": "rules", "kev_url": "http://127.0.0.1:8008",
                    "kev_model": "kev-latest", "kev_api_key": "", "reflex_threshold": 0.55,
                    "refuse_threshold": 0.2, "timeout_ms": 1500 },
"senses": { "vision_enabled": true, "screenshot_enabled": false, "vision_base_url": "",
            "vision_model": "", "vision_api_key": "", "hearing_enabled": true,
            "hearing_window_secs": 180, "hearing_max_lines": 8 },
"voice_direction": { "enabled": false, "vocal_events": true, "radio_enabled": true,
                     "radio_distance_m": 25.0, "radio_static": 0.04 }
```

`profile.json` can carry a `reflexes` array (`id`, `description`, `phrases`,
`needs_target`) and `features.voiceDirection`. Without `reflexes`, a generic set
is used (`jump`, `crouch`, `follow`, `wait`, `move_to`, `hold_fire`, …).
