"""halo_sim.py - a fake Halo CE bridge for exercising chasm without the game.

It plays the game side of the bridge contract (docs/halo-ce-bridge.md): it
reports what Kevin can see (landmarks with coordinates), feeds him overheard
radio chatter, sends the player's lines, and prints what chasm sends back -
nervous-system reflexes, brain replies, actions, and audio chunks.

Scripted mode (default) replays beats from the "Kevin" demo and checks each
result, so it doubles as an integration test:

    python scripts/halo_sim.py                      # nervous system / senses only (no LLM needed)
    python scripts/halo_sim.py --with-brain         # also run full LLM turns
    python scripts/halo_sim.py --interactive        # type lines to Kevin yourself
    python scripts/halo_sim.py --save-audio out/    # keep the WAV chunks

Stdlib only. The chasm backend must be running with the halo-ce profile active.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

LANDMARKS = [
    {"id": "rock_01", "label": "large boulder", "kind": "cover", "distance_m": 8.0,
     "direction": "ahead-left", "position": {"x": 112.4, "y": -36.0, "z": 4.1}},
    {"id": "bodies_01", "label": "pile of dead Marines", "kind": "corpse", "distance_m": 22.0,
     "direction": "right", "position": {"x": 131.0, "y": -12.5, "z": 3.2}},
    {"id": "lifeboat_01", "label": "crashed lifeboat", "kind": "wreck", "distance_m": 35.0,
     "direction": "ahead", "position": {"x": 150.2, "y": -40.8, "z": 2.0}},
    {"id": "hog_01", "label": "Warthog", "kind": "vehicle", "distance_m": 12.0,
     "direction": "behind", "position": {"x": 96.0, "y": -30.1, "z": 3.9}},
]
OBJECTIVE = "Evade the Covenant patrols by heading into the hills, then scout for the other lifeboats."


class Chasm:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def turn(self, body: dict):
        """Yields each NDJSON event of a streamed turn as it arrives."""
        request = urllib.request.Request(
            self.base + "/api/game/v1/turn", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    yield json.loads(line)


def kevin_npc(distance: float) -> dict:
    return {"npc_key": "kevin", "npc_name": "Kevin", "characterId": "Kevin",
            "distance_m": distance, "under_crosshair": distance < 15}


def turn_body(text: str, distance: float = 3.0, *, in_combat: bool = False, heard=None) -> dict:
    return {
        "npc_key": "kevin",
        "npc_name": "Kevin",
        "player_text": text,
        "want_tts": True,
        "location": {"major": "Halo", "minor": "Lifeboat crash site"},
        "metadata": {
            "targeting": {"nearby_npcs": [kevin_npc(distance)]},
            "vision": {"landmarks": LANDMARKS},
            "objective": OBJECTIVE,
            "radio_capable": True,
            "in_combat": in_combat,
            "heard": heard or [],
        },
    }


def stimulus_body(text: str) -> dict:
    return {"npc_name": "Kevin", "kind": "speech", "speaker": "player", "text": text,
            "landmarks": LANDMARKS, "situation": f"Location: Halo. Current objective: {OBJECTIVE}"}


class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label: str, ok: bool, detail: str = "") -> None:
        mark = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{mark}] {label}" + (f" - {detail}" if detail else ""))


def describe(decision: dict) -> str:
    reflex = decision.get("reflex") or {}
    target = (reflex.get("target") or {}).get("label")
    parts = [decision.get("layer", "?")]
    if reflex:
        parts.append(reflex.get("id", "?") + (f" -> {target}" if target else ""))
    if decision.get("refused"):
        parts.append("REFUSED")
    parts.append(f"{decision.get('source')} {decision.get('latency_ms')}ms")
    return " | ".join(parts)


def run_turn(chasm: Chasm, body: dict, save_audio: str | None, label: str) -> list[dict]:
    events = []
    started = time.monotonic()
    first_audio = None
    for event in chasm.turn(body):
        events.append(event)
        kind = event.get("type")
        elapsed = int((time.monotonic() - started) * 1000)
        if kind == "nervous":
            print(f"    {elapsed:5d}ms nervous: {describe(event['decision'])}")
        elif kind == "reflex":
            target = (event.get("target") or {}).get("label")
            print(f"    {elapsed:5d}ms REFLEX {event['action']}" + (f" -> {target}" if target else ""))
        elif kind == "speech.delta":
            print(f"    {elapsed:5d}ms Kevin: {event.get('text')}")
        elif kind == "audio.chunk":
            first_audio = first_audio if first_audio is not None else elapsed
            extra = []
            if event.get("delivery"):
                extra.append(f"delivery={event['delivery']}")
            if event.get("channel"):
                extra.append(f"channel={event['channel']}")
            print(f"    {elapsed:5d}ms audio chunk #{event['index']} {' '.join(extra)}")
            if save_audio:
                os.makedirs(save_audio, exist_ok=True)
                path = os.path.join(save_audio, f"{label}_{event['index']:03d}.wav")
                with open(path, "wb") as handle:
                    handle.write(base64.b64decode(event["audio"]["data"]))
        elif kind == "action":
            target = event.get("target")
            where = (event.get("resolvedTarget") or {}).get("position")
            print(f"    {elapsed:5d}ms ACTION {event.get('actionId')}"
                  + (f" target={target!r}" if target else "") + (f" at {where}" if where else ""))
        elif kind == "reply":
            print(f"    {elapsed:5d}ms reply: {event.get('text')!r}")
        elif kind == "turn.error":
            print(f"    {elapsed:5d}ms ERROR: {event.get('error')}")
        elif kind == "turn.completed":
            print(f"    {elapsed:5d}ms done")
    return events


def scripted(chasm: Chasm, with_brain: bool, save_audio: str | None) -> int:
    report = Report()

    print("\n== Hearing: Cortana over the radio")
    heard = chasm.post("/api/game/v1/hear", {
        "speaker": "Cortana", "channel": "radio", "listeners": ["Kevin"],
        "text": "They're trying to flank us. Watch your six.", "react": True})
    report.check("radio line recorded", heard.get("recorded") == 1, json.dumps(heard)[:160])

    print("\n== Nervous system (no LLM)")
    beats = [
        ("Kevin, jump.", "reflex", "body.jump", None),
        ("Kevin, can you jump?", "brain", None, None),
        ("Kevin, crouch.", "reflex", "body.crouch", None),
        ("Kevin, stand up.", "reflex", "body.stand", None),
        ("Kevin, go stand near those dead Marines over there.", "reflex", "squad.move_to", "bodies_01"),
        ("Stand by this boulder.", "reflex", "squad.move_to", "rock_01"),
        ("Okay Kevin, I want you to hold your fire until I say so.", "reflex", "combat.hold_fire", None),
        ("Shoot!", "reflex", "combat.fire_at_will", None),
        ("Kevin, I want you to wait here rather than going in.", "reflex", "squad.wait", None),
        ("Get in the car, Kevin.", "reflex", "vehicle.enter", None),
        ("Kevin, what's your opinion on Grunts?", "brain", None, None),
    ]
    for text, layer, reflex_id, target_id in beats:
        decision = chasm.post("/api/game/v1/stimulus", stimulus_body(text))
        reflex = decision.get("reflex") or {}
        ok = decision.get("layer") == layer and reflex.get("id") == reflex_id
        if target_id:
            ok = ok and (reflex.get("target") or {}).get("id") == target_id
        report.check(f"{text!r} -> {layer} {reflex_id or ''}".rstrip(), ok, describe(decision))

    print("\n== Eyes: /look")
    look = chasm.post("/api/game/v1/look", {"npc_name": "Kevin", "text": "that Warthog behind us",
                                            "landmarks": LANDMARKS})
    report.check("'that Warthog behind us' -> hog_01",
                 (look.get("target") or {}).get("id") == "hog_01", json.dumps(look)[:160])

    print("\n== Reflex-only turn (instant, the brain never runs)")
    events = run_turn(chasm, turn_body("Kevin, jump."), save_audio, "jump")
    types = [e.get("type") for e in events]
    report.check("turn streams nervous -> reflex -> reply -> completed",
                 types[:2] == ["nervous", "reflex"] and types[-1] == "turn.completed"
                 and "audio.chunk" not in types, str(types))

    if with_brain:
        print("\n== Brain turn: mission awareness + what he heard")
        events = run_turn(chasm, turn_body("Kevin, what are we supposed to be doing right now?"),
                          save_audio, "mission")
        report.check("brain turn replied", any(e.get("type") == "reply" and e.get("text") for e in events))

        print("\n== Reflex + reply: he jumps, then talks")
        events = run_turn(chasm, turn_body("Kevin, jump. Come on, show me what a Spartan can do."),
                          save_audio, "jump_talk")
        types = [e.get("type") for e in events]
        report.check("reflex fired before the brain spoke",
                     "reflex" in types and "reply" in types and types.index("reflex") < types.index("reply"),
                     str(types))

        print("\n== Radio: Kevin is 60m away")
        events = run_turn(chasm, turn_body("Kevin, where are you?", distance=60.0), save_audio, "radio")
        chunks = [e for e in events if e.get("type") == "audio.chunk"]
        report.check("far-away reply comes over the radio",
                     bool(chunks) and all(c.get("channel") == "radio" for c in chunks),
                     f"{len(chunks)} chunk(s)")

    print(f"\n{report.passed} passed, {report.failed} failed")
    return 1 if report.failed else 0


def interactive(chasm: Chasm, save_audio: str | None) -> int:
    print("Talk to Kevin. Prefix with '/far ' to talk over the radio, '/hear ' to make him overhear "
          "Cortana, '/combat ' for an in-combat line. Ctrl-D to quit.")
    n = 0
    while True:
        try:
            line = input("\nyou> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line.startswith("/hear "):
            chasm.post("/api/game/v1/hear", {"speaker": "Cortana", "channel": "radio",
                                              "listeners": ["Kevin"], "text": line[6:]})
            print("    (Kevin heard it)")
            continue
        distance, combat = 3.0, False
        if line.startswith("/far "):
            distance, line = 60.0, line[5:]
        if line.startswith("/combat "):
            combat, line = True, line[8:]
        n += 1
        try:
            run_turn(chasm, turn_body(line, distance, in_combat=combat), save_audio, f"turn{n:03d}")
        except urllib.error.URLError as error:
            print(f"    request failed: {error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=os.environ.get("CHASM_URL", "http://127.0.0.1:7341"))
    parser.add_argument("--with-brain", action="store_true", help="also run full LLM turns")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--save-audio", metavar="DIR")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    chasm = Chasm(args.base, args.timeout)
    try:
        with urllib.request.urlopen(chasm.base + "/health", timeout=5) as response:
            json.loads(response.read())
    except (urllib.error.URLError, OSError) as error:
        print(f"chasm is not reachable at {chasm.base}: {error}", file=sys.stderr)
        return 2
    if args.interactive:
        return interactive(chasm, args.save_audio)
    return scripted(chasm, args.with_brain, args.save_audio)


if __name__ == "__main__":
    sys.exit(main())
