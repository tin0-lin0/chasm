# CLAUDE.md: notes for AI agents working on this fork

This is a fork of [chasmlol/chasm](https://github.com/chasmlol/chasm), a Rust
(axum) + React engine for LLM-driven game NPCs. The fork adds the features from
chasm's demo video "Giving Video Game NPC's Consciousness"
(<https://www.youtube.com/watch?v=EOfm3h4xYU8>): **Kevin**, an AI squadmate in
Halo: Combat Evolved (the OpenCE decompilation).

The models in the video are Kev-4B (nervous system), Gemma 4 12B (brain),
Parakeet (STT) and Breeze TTS 2 (directed voice).

## Git setup

- `origin` is the fork, `tin0-lin0/chasm`. `upstream` is `chasmlol/chasm`. Never push to `upstream`.
- The work lives on the branch **`halo-kevin-features`**. The owner chose to keep it
  off `main` for now, so don't merge into `main` unless they ask.
- To pick up upstream changes, run `git fetch upstream && git merge upstream/main`
  on the feature branch.

## Scope the owner chose

- **Halo side: a profile and a simulator only.** There is no in-game OpenCE C mod.
  `scripts/halo_sim.py` plays the game side of the bridge contract. The game-side
  hooks are only *suggested*, in `docs/halo-ce-bridge.md` §4.
- **Features built:**
  - Nervous system (Kev-4B with a rules fallback).
  - Eyes and hearing (landmarks, screenshot description, overheard lines).
  - Voice direction and a radio effect.
- **Not built:** the "mind console + orders" UI from the video. The owner did not pick it.

## Where the new code is

The full contract is in `docs/halo-ce-bridge.md`. Read it first.

| Area | Files |
| --- | --- |
| Settings | `crates/chasm-core/src/settings.rs`: `NervousSystemSettings`, `SensesSettings`, `VoiceDirectionSettings`, and the `breeze-tts-2` local engine id |
| Nervous system | `crates/chasm-web/src/nervous.rs`: `decide()` calls Kev's `/v1/systemone` and falls back to `decide_rules`. Its tests use a mock Kev server. |
| Senses | `crates/chasm-web/src/senses.rs`: landmarks, the hearing store (a process-global `OnceLock<Mutex<VecDeque>>`), `/hear`, `/look`, the screenshot→vision call, and the prompt components |
| Radio FX | `crates/chasm-web/src/radio.rs`: band-pass, saturation, static and squelch on 16-bit WAV chunks |
| Voice direction | `crates/chasm-fnv-bridge/src/delivery.rs`: `[cue]` / `(vocal event)` splitting and `CueTracker` |
| Turn pipeline | `crates/chasm-web/src/game_bridge.rs` `turn()`: inline heard lines → screenshot → radio decision → nervous pass (`nervous`/`reflex` events) → brain |
| Prompt | `crates/chasm-web/src/generate.rs`: new components. Per-turn ones must be in `VOLATILE_KEYS` to keep the prompt-cache head stable. |
| Routes | `crates/chasm-web/src/lib.rs`: `/api/game/v1/{stimulus,hear,look}`, `directed_speech()`, `BREEZE_TTS_ENGINE` |
| Action trust | `crates/chasm-fnv-bridge/src/actions.rs`: `<game>:http-bridge` bindings are trusted like FNV's xNVSE |
| Breeze | `scripts/breeze_tts_server.py`: a stdlib adapter that turns chasm's JSON TTS request into Breeze's multipart API. `launcher.rs` `build_breeze_spec` starts it. |
| Halo profile | `profiles/halo-ce/`: `profile.json` (reflexes, `features.voiceDirection`), the Kevin card source and `build_cards.py`, the lorebook, and the action book |
| Simulator | `scripts/halo_sim.py`: scripted checks, `--with-brain`, `--interactive`, `--save-audio` |

## Build and test

```bash
cargo test --workspace          # everything should pass
cargo clippy --workspace
cargo build -p chasm            # the server binary (apps/chasm)
```

- **MSRV is 1.80.** Clippy flags newer std APIs; for example, use `map_or(true, …)`, not `Option::is_none_or`.
- **Upstream is not rustfmt-clean.** Run `rustfmt` only on files you touch, not
  `cargo fmt --all`, or the diff will pick up unrelated reformatting.
- **Tests and the hearing store.** The store is global. A test that records heard
  lines must scope them to its own listener names, or it leaks into other tests,
  e.g. `warmup_prompt_is_the_real_first_turn_prefix_and_persists_nothing`.
- **Test artifact.** Tests can leave a `crates/chasm-web/NVBridgeStream/` directory behind. Delete it; don't commit it.
- **Python scripts are stdlib-only.** Python 3.13 has no `cgi` module, so use `email.parser` for multipart.
- **Target platform.** The repo targets Windows (the launcher, `.ps1` scripts and the Tauri installer). Development so far was on macOS.

## Running end to end without the game

Start the server with the `halo-ce` profile active, then run the simulator against it:

```bash
CHASM_DATA_ROOT=… CHASM_SETTINGS=…/chasm.settings.json CHASM_PROFILES_DIR=…/profiles \
CHASM_VOICES_DIR=… CHASM_BIND_ADDR=127.0.0.1:17400 \
CHASM_LLM_ENDPOINT=http://127.0.0.1:<llm> CHASM_TTS_ENDPOINT=http://127.0.0.1:<tts> \
  target/debug/chasm
python3 scripts/halo_sim.py --base http://127.0.0.1:17400               # nervous system + senses
python3 scripts/halo_sim.py --base http://127.0.0.1:17400 --with-brain  # plus LLM turns
```

Use a sandbox data dir like this so you don't touch the owner's real chasm data.
The earlier verification used throwaway stand-ins that are **not in the repo**: a
mock OpenAI-compatible LLM, a fake Breeze and a fake STT. Write new ones or point
at real models.

## Status

**Verified**
- `cargo test --workspace` passes.
- `halo_sim.py` passed 17/17 checks against a sandboxed server with a mock LLM and fake Breeze/STT.
- The LLM prompts contained the mission, vision, overheard, instinct and radio blocks.
- Breeze received the voice-direction instruction, and vocal events were kept.

**Not verified**
- Real Kev-4B, real Breeze TTS 2, and Gemma 4 12B.
- The Windows launcher path for Breeze.
- Any real game.

**Not built**
- A Settings UI page for the new `nervous_system`, `senses` and `voice_direction`
  settings. Today they are set in `chasm.settings.json` or the profile.

## Open threads and next steps

1. **Voice direction on hosted TTS.** `directed_speech()` strips cues for hosted
   providers, so direction only works with Breeze.
   - The proposed fix: add ElevenLabs `eleven_v3` to the model list, and send cues
     and vocal events as v3 audio tags (`[whispers]`, `[laughs]`) when that model is selected.
   - This gives directed voice on any hardware.
   - The owner hasn't decided yet.
2. **Hardware.** Breeze TTS 2 is documented for Linux and NVIDIA, but it isn't hard-limited to that.
   - The default path is eager attention, bf16 and plain torch, and it falls back to the CPU, which is too slow for live use.
   - AMD via ROCm on Linux is plausible. Native Windows on NVIDIA is plausible.
   - Neither is tested. There is no device flag, so Mac would need a patch.
   - The Breeze weights are licensed **non-commercial**.
3. **Test with real Kev-4B** (an ~8 GB download). See `docs/halo-ce-bridge.md` §6 step 3.
4. **The OpenCE game-side mod** is out of scope for now. §4 of the doc lists the suggested engine hooks.
5. **The mind console and orders UI** from the video were not requested yet.

## Conventions

- Match the surrounding code's comment density and naming. New modules carry a `//!` header that explains the why.
- New game-facing behavior goes in `docs/halo-ce-bridge.md`, and new turn metadata and NDJSON events go in its tables.
- Keep new features **additive and off by default** where they change the upstream
  FNV behavior. Halo-specific behavior is switched on by the `halo-ce` profile.
- License: AGPL-3.0 (the workspace). Game data and voices are not included.
