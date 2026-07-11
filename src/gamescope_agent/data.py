"""Loading and validation of the local demonstration corpus."""

from __future__ import annotations

import json
from pathlib import Path

from gamescope_agent.models import GameRecord


def load_game_records(data_dir: Path) -> list[GameRecord]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Game data directory does not exist: {data_dir}")

    files = sorted(data_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON records found in: {data_dir}")

    records: list[GameRecord] = []
    for file_path in files:
        with file_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["SourceFile"] = file_path.name
        records.append(GameRecord.model_validate(payload))

    return records
