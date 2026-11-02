"""chasm adapter for Breeze TTS 2 (voice clone + voice direction + vocal events).

Breeze TTS 2 ships its own streaming API (`python -m breeze_infer.api <model>
--port 7860`, Linux + CUDA; run it natively, in WSL2, or from its Docker image).
This adapter puts chasm's local-TTS contract in front of it, so chasm can use
Breeze exactly like faster-qwen3-tts / PocketTTS on :5002:

  GET  /health
  POST /v1/audio/speech   {input, voice, instructions?, response_format: "pcm",
                           lead_in_ms?, trailing_ms?, ...}
       -> streaming mono 24 kHz s16le PCM

Per request it resolves the character's reference clip
(`<voices-dir>/<voice>/breeze-tts-2/prompt.wav`, else `<voice>/reference.wav`)
and its EXACT transcript, which Breeze needs for cloning:

  1. `<voices-dir>/<voice>/reference.txt` (hand-written), else
  2. `<voices-dir>/<voice>/breeze-tts-2/reference.txt` (cached), else
  3. transcribed once through chasm's own STT (`--chasm-url`) and cached.

`instructions` (the line's delivery cue, e.g. "whispering", "thick German
accent") switches Breeze from plain cloning to Voice Direction with
`cfg_scale` 4. Inline vocal events in the text — "(laugh)", "(sigh)" — are
performed by the model. With no usable clip, the adapter falls back to Voice
Design from `--default-voice-design`.

Stdlib only: runs on any Python 3.9+, including on Windows next to a Breeze
server that runs in WSL2.

  python breeze_tts_server.py --voices-dir C:/.../voices --port 5002 \
      --breeze-url http://127.0.0.1:7860 --chasm-url http://127.0.0.1:7341
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

SAMPLE_RATE = 24_000
ENGINE_ID = "breeze-tts-2"

ARGS: argparse.Namespace
_transcript_lock = threading.Lock()


def log(message: str) -> None:
    print(f"[breeze-adapter] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

def ref_audio_path(voice: str) -> Optional[str]:
    if not ARGS.voices_dir or not voice:
        return None
    for candidate in (
        os.path.join(ARGS.voices_dir, voice, ENGINE_ID, "prompt.wav"),
        os.path.join(ARGS.voices_dir, voice, "reference.wav"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def first_voice() -> Optional[str]:
    if not ARGS.voices_dir or not os.path.isdir(ARGS.voices_dir):
        return None
    for name in sorted(os.listdir(ARGS.voices_dir)):
        if ref_audio_path(name):
            return name
    return None


def read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read().strip()
        return text or None
    except OSError:
        return None


def transcribe_with_chasm(wav_path: str) -> Optional[str]:
    """Asks chasm's own STT for the clip's transcript (Parakeet or a hosted key)."""
    if not ARGS.chasm_url:
        return None
    with open(wav_path, "rb") as handle:
        audio = base64.b64encode(handle.read()).decode("ascii")
    body = json.dumps({
        "audio": {"data": audio, "encoding": "base64", "format": "wav", "mimeType": "audio/wav"},
    }).encode("utf-8")
    url = ARGS.chasm_url.rstrip("/") + "/api/headless/v1/speech/recognize"
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as error:
        log(f"transcribe {wav_path}: {error}")
        return None
    for key in ("text", "transcript"):
        if isinstance(value.get(key), str) and value[key].strip():
            return value[key].strip()
    result = value.get("result") or {}
    if isinstance(result, dict) and isinstance(result.get("text"), str):
        return result["text"].strip() or None
    return None


def ref_transcript(voice: str, wav_path: str) -> Optional[str]:
    manual = read_text(os.path.join(ARGS.voices_dir, voice, "reference.txt"))
    if manual:
        return manual
    cache = os.path.join(ARGS.voices_dir, voice, ENGINE_ID, "reference.txt")
    cached = read_text(cache)
    if cached:
        return cached
    with _transcript_lock:
        cached = read_text(cache)
        if cached:
            return cached
        text = transcribe_with_chasm(wav_path)
        if text:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            with open(cache, "w", encoding="utf-8") as handle:
                handle.write(text)
            log(f"cached transcript for {voice!r}: {text[:60]!r}")
        return text


# ---------------------------------------------------------------------------
# Breeze request
# ---------------------------------------------------------------------------

def multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode("utf-8")
            + str(value).encode("utf-8") + b"\r\n"
        )
    for name, path in files.items():
        with open(path, "rb") as handle:
            data = handle.read()
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
            f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8")
            + data + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def breeze_fields(req: dict) -> tuple[dict, dict]:
    text = str(req.get("input") or req.get("text") or "").strip()
    voice = str(req.get("voice") or "").strip() or (first_voice() or "")
    instruction = str(req.get("instructions") or req.get("instruction") or "").strip()
    fields: dict = {"text": text}
    files: dict = {}
    wav = ref_audio_path(voice)
    transcript = ref_transcript(voice, wav) if wav else None
    if wav and transcript:
        files["ref_audio"] = wav
        fields["ref_text"] = transcript
        if instruction:
            fields["instruction"] = instruction
            fields["cfg_scale"] = ARGS.cfg_scale
    else:
        if wav:
            log(f"no transcript for {voice!r}; using voice design")
        design = ARGS.default_voice_design
        fields["instruction"] = f"{design} {instruction}".strip() if instruction else design
        fields["cfg_scale"] = ARGS.cfg_scale
    if ARGS.seed is not None:
        fields["seed"] = ARGS.seed
    return fields, files


def open_breeze_stream(req: dict):
    """POSTs to Breeze, retrying while its single inference slot is busy (409)."""
    fields, files = breeze_fields(req)
    body, content_type = multipart(fields, files)
    url = ARGS.breeze_url.rstrip("/") + "/v1/audio/speech"
    deadline = time.monotonic() + ARGS.busy_wait_s
    while True:
        request = urllib.request.Request(url, data=body, headers={"Content-Type": content_type})
        try:
            return urllib.request.urlopen(request, timeout=ARGS.timeout_s)
        except urllib.error.HTTPError as error:
            if error.code == 409 and time.monotonic() < deadline:
                time.sleep(0.15)
                continue
            raise


def silence(ms: int) -> bytes:
    return b"\x00\x00" * max(0, int(SAMPLE_RATE * ms / 1000))


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter default logging
        pass

    def send_json(self, status: int, value: dict) -> None:
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_chunk(self, data: bytes) -> None:
        if data:
            self.wfile.write(f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n")
            self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") != "/health":
            self.send_json(404, {"error": "not found"})
            return
        upstream = False
        try:
            with urllib.request.urlopen(ARGS.breeze_url.rstrip("/") + "/v1/model", timeout=2) as r:
                upstream = r.status == 200
        except (urllib.error.URLError, OSError):
            upstream = False
        self.send_json(200, {"status": "ok", "engine": ENGINE_ID, "model_loaded": upstream})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/audio/speech":
            self.send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self.send_json(400, {"error": "invalid JSON"})
            return
        if not str(req.get("input") or req.get("text") or "").strip():
            self.send_json(400, {"error": "input is required"})
            return
        if str(req.get("response_format") or "pcm").lower() != "pcm":
            self.send_json(400, {"error": "only response_format=pcm is supported"})
            return
        try:
            upstream = open_breeze_stream(req)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            log(f"breeze status {error.code}: {detail}")
            self.send_json(502, {"error": f"Breeze status {error.code}", "detail": detail})
            return
        except (urllib.error.URLError, OSError) as error:
            log(f"breeze unreachable at {ARGS.breeze_url}: {error}")
            self.send_json(503, {"error": f"Breeze unreachable: {error}"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        try:
            self.write_chunk(silence(int(req.get("lead_in_ms") or 0)))
            carry = b""
            with upstream:
                while True:
                    data = upstream.read(4800)
                    if not data:
                        break
                    data = carry + data
                    # Keep sample alignment across reads.
                    cut = len(data) - (len(data) % 2)
                    carry = data[cut:]
                    self.write_chunk(data[:cut])
            self.write_chunk(silence(int(req.get("trailing_ms") or 0)))
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # chasm cancelled the line (speaker died / client hung up).
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="chasm adapter for Breeze TTS 2")
    p.add_argument("--voices-dir", default=os.environ.get("BREEZE_VOICES_DIR", ""))
    p.add_argument("--breeze-url", default=os.environ.get("CHASM_BREEZE_URL", "http://127.0.0.1:7860"))
    p.add_argument("--chasm-url", default=os.environ.get("CHASM_URL", "http://127.0.0.1:7341"),
                   help="chasm backend, used to transcribe reference clips once")
    p.add_argument("--cfg-scale", type=float, default=4.0,
                   help="CFG for voice direction / design (Breeze recommends 4)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--default-voice-design",
                   default="A gruff, battle-hardened soldier with a rough, low voice.")
    p.add_argument("--busy-wait-s", type=float, default=20.0,
                   help="how long to retry while Breeze's single slot is busy")
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5002)
    return p.parse_args()


def main() -> None:
    global ARGS
    ARGS = parse_args()
    server = ThreadingHTTPServer((ARGS.host, ARGS.port), Handler)
    log(f"listening on http://{ARGS.host}:{ARGS.port} -> Breeze {ARGS.breeze_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
