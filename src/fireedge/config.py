from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the thermal-edge-fire project root."""
    # This file is located at <root>/src/fireedge/config.py
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load configuration YAML file."""
    if config_path is None:
        config_path = get_project_root() / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def resolve_path(rel_or_abs_path: str | Path) -> Path:
    """Resolve a path relative to project root if not absolute."""
    path = Path(rel_or_abs_path)
    if path.is_absolute():
        return path
    return get_project_root() / path


def get_data_dir(key: str = "activefire_root") -> Path:
    """Retrieve and ensure data directory exists."""
    cfg = load_config()
    data_cfg = cfg.get("data", {})
    rel_path = data_cfg.get(key, f"data/raw/{key}")
    full_path = resolve_path(rel_path)
    full_path.mkdir(parents=True, exist_ok=True)
    return full_path


def get_reports_dir() -> Path:
    """Retrieve and ensure reports directory exists."""
    cfg = load_config()
    rel_path = cfg.get("data", {}).get("reports_dir", "reports")
    full_path = resolve_path(rel_path)
    full_path.mkdir(parents=True, exist_ok=True)
    (full_path / "figures").mkdir(parents=True, exist_ok=True)
    return full_path
