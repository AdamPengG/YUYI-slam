from __future__ import annotations

import json
from pathlib import Path

from ..data_types import KeyframePacket


def append_keyframe_packet(path: Path, packet: KeyframePacket) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet.to_dict(), ensure_ascii=False) + "\n")


def load_keyframe_packets(path: Path) -> list[KeyframePacket]:
    if not path.exists():
        return []
    packets: list[KeyframePacket] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        packets.append(KeyframePacket.from_dict(json.loads(line)))
    return packets
