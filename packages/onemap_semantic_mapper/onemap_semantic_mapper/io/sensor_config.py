from __future__ import annotations

from pathlib import Path

import yaml

from ..data_types import SensorConfig


def write_sensor_config(path: Path, config: SensorConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def load_sensor_config(path: Path) -> SensorConfig:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return SensorConfig.from_dict(payload)
