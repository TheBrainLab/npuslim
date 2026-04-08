"""Runtime bootstrap for CLI: config parse/validate/logging/presentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from npuslim.config.parser import parse_config
from npuslim.config.printer import (
    dump_config_snapshot,
    print_config,
    resolve_log_dir,
    setup_logger,
    show_npuslim_header,
)
from npuslim.config.schema import EngineConfig
from npuslim.config.validator import validate_config


def _load_raw_yaml(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Top-level YAML must be a mapping, got: {type(data).__name__}"
        )
    return data


def _relative_config_stem(cfg_path: Path) -> Path:
    cfg_abs = cfg_path.resolve()
    try:
        rel = cfg_abs.relative_to(Path.cwd())
    except ValueError:
        rel = Path(cfg_abs.name)
    stem = rel.with_suffix("")
    parts = list(stem.parts)
    for idx, part in enumerate(parts):
        if part.lower() in {"config", "configs", "cfg", "cfgs"} and idx + 1 < len(
            parts
        ):
            return Path(*parts[idx + 1 :])
    return stem


def apply_saver_path_policy(config: EngineConfig, cfg_path: Path) -> None:
    """
    Resolve saver output paths from config path.

    Policy:
    - If saver.save_path is present, keep it as-is (highest priority).
    - Else if saver.save_dir or saver.output_dir exists, treat it as root and
      derive saver.save_path = <root>/<relative-config-stem>.
    """
    rel_stem = _relative_config_stem(cfg_path)

    for task in config.recipe:
        saver = task.saver
        if not isinstance(saver, dict):
            continue

        explicit_save_path = saver.get("save_path")
        if explicit_save_path:
            continue

        root = saver.get("save_dir") or saver.get("output_dir")
        if not root:
            continue

        resolved_save_path = Path(str(root)) / rel_stem
        saver["save_path"] = str(resolved_save_path)


def bootstrap_from_path(
    cfg_path: Path,
    *,
    log_dir: Optional[str] = None,
    show_header: bool = True,
    strict_validate: bool = True,
) -> EngineConfig:
    """Bootstrap runtime from config path and return parsed EngineConfig."""
    raw_cfg = _load_raw_yaml(cfg_path)
    resolved_log_dir = resolve_log_dir(cfg_path, raw_cfg, override=log_dir)
    setup_logger(log_dir=resolved_log_dir)
    dump_config_snapshot(raw_cfg, resolved_log_dir)

    parsed_cfg = parse_config(cfg_path)
    apply_saver_path_policy(parsed_cfg, cfg_path)
    validate_config(parsed_cfg, strict=strict_validate)
    print_config(parsed_cfg, title=f"Configuration of {cfg_path}")
    if show_header:
        show_npuslim_header()
    return parsed_cfg
