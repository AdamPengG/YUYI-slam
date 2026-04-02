from __future__ import annotations

import json
from pathlib import Path

from ..data_types import LocalCloudPacket


def append_local_cloud_packet(path: Path, packet: LocalCloudPacket) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet.to_dict(), ensure_ascii=False) + "\n")


def load_local_cloud_packets(path: Path) -> list[LocalCloudPacket]:
    if not path.exists():
        return []
    packets: list[LocalCloudPacket] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        packets.append(LocalCloudPacket.from_dict(json.loads(line)))
    return packets
