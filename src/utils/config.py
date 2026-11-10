"""Load project configuration and resolve paths from the project root."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "experiment_config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open() as handle:
        config = yaml.safe_load(handle)
    config["_root"] = ROOT
    return config


def project_path(config: dict, relative: str) -> Path:
    path = config["_root"] / relative
    path.mkdir(parents=True, exist_ok=True)
    return path


def path_of(config: dict, key: str) -> Path:
    return project_path(config, config["paths"][key])


def append_event(config: dict, stage: str, message: str, **extra) -> None:
    logs = path_of(config, "logs_dir")
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "message": message,
        **extra,
    }
    with (logs / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
