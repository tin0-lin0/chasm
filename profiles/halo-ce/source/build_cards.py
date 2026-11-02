"""Builds characters/<Name>.png character cards from source/<Name>.card.json.

chasm reads character cards as PNGs with the card JSON embedded (base64) in a
`chara` tEXt chunk (SillyTavern V2 layout). This writes a small solid-color
portrait carrying the card, so the card stays editable as JSON in the repo.

    python profiles/halo-ce/source/build_cards.py
"""

import base64
import json
import pathlib
import struct
import zlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "characters"


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def portrait(width: int = 64, height: int = 64, rgb=(74, 96, 58)) -> bytes:
    row = b"\x00" + bytes(rgb) * width
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9))


def build(card_path: pathlib.Path) -> pathlib.Path:
    card = json.loads(card_path.read_text(encoding="utf-8"))
    name = card["data"]["name"]
    payload = base64.b64encode(json.dumps(card, ensure_ascii=False).encode("utf-8"))
    png = b"\x89PNG\r\n\x1a\n" + portrait() + chunk(b"tEXt", b"chara\x00" + payload) + chunk(b"IEND", b"")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{name}.png"
    target.write_bytes(png)
    return target


if __name__ == "__main__":
    for path in sorted(HERE.glob("*.card.json")):
        print(f"wrote {build(path)}")
