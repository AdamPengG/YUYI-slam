from __future__ import annotations

import json
from pathlib import Path

from ..data_types import ObservationLink


def append_observation_link(path: Path, observation: ObservationLink) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")


def load_observation_links(path: Path) -> list[ObservationLink]:
    if not path.exists():
        return []
    observations: list[ObservationLink] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        observations.append(ObservationLink.from_dict(json.loads(line)))
    return observations
