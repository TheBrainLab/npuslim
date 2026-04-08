"""Config/logging presentation utilities for the v2 runtime."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.text import Text

from npuslim.config.schema import EngineConfig


def setup_logger(
    *,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
) -> Optional[Path]:
    """Configure loguru console/file outputs in a v1-like style."""
    import sys

    def _patch_short_file(record):
        file_name = record["file"].name
        max_len = 16
        if len(file_name) > max_len:
            file_name = file_name[: max_len - 2] + ".."
        record["extra"]["short_file"] = f"{file_name:<{max_len}}"

    logger.remove()
    logger.configure(patcher=_patch_short_file)

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[short_file]}:{line: <4}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        level=level,
        format=log_format,
        filter=lambda record: "quiet" not in record["extra"],
    )

    log_path: Optional[Path] = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{timestamp}.log"
        logger.add(
            str(log_path),
            level=level,
            format=log_format,
            rotation="500 MB",
            encoding="utf-8",
            enqueue=False,
        )

    logger.opt(colors=True).info(
        "<blue>╔══════════════════════════════════════════════════════════╗</blue>"
    )
    logger.opt(colors=True).info(
        "<blue>║</blue>      NPUSlim Toolchain Initialized                       <blue>║</blue>"
    )
    logger.opt(colors=True).info(
        "<blue>╚══════════════════════════════════════════════════════════╝</blue>"
    )
    if log_path is not None:
        logger.info(f"Log file: {log_path}")
    return log_path


def show_npuslim_header() -> None:
    """Print NPUSlim banner."""
    try:
        from npuslim import __version__
    except Exception:
        __version__ = "0.0.0-dev"

    logo_lines = [
        "<blue>███╗  ██╗██████╗ ██╗   ██╗███████╗██╗     ██╗███╗   ███╗</blue>",
        "<blue>████╗ ██║██╔══██╗██║   ██║██╔════╝██║     ██║████╗ ████║</blue>",
        "<blue>██╔██╗██║██████╔╝██║   ██║███████╗██║     ██║██╔████╔██║</blue>",
        "<blue>██║╚████║██╔═══╝ ██║   ██║╚════██║██║     ██║██║╚██╔╝██║</blue>",
        "<blue>██║ ╚███║██║     ╚██████╔╝███████║███████╗██║██║ ╚═╝ ██║</blue>",
    ]
    description = "Unified Compression, Acceleration & Deployment Toolchain for Ascend NPU"

    logger.opt(colors=True, raw=True).info("<blue>" + "━" * 80 + "</blue>\n")
    for line in logo_lines:
        logger.opt(colors=True, raw=True).info(f"{line}\n")
    logger.opt(colors=True, raw=True).info(
        f"\n<italic><blue>{description}</blue></italic>\n"
    )
    logger.opt(colors=True, raw=True).info(
        f"<white><bold>version</bold></white>  <bg white><black> {__version__} </black></bg white>\n"
    )
    logger.opt(colors=True, raw=True).info("<blue>" + "━" * 80 + "</blue>\n")


def resolve_log_dir(
    cfg_path: Path,
    raw_config: Dict[str, Any],
    *,
    override: Optional[str] = None,
) -> Path:
    """Resolve log dir from CLI override, config metadata, or default path."""
    if override:
        return Path(override)

    metadata = raw_config.get("metadata", {}) or {}
    configured = metadata.get("log_dir") or metadata.get("work_dir")
    if configured:
        return Path(str(configured))

    cfg_abs = cfg_path.resolve()
    try:
        rel = cfg_abs.relative_to(Path.cwd())
    except ValueError:
        rel = Path(cfg_abs.name)
    return Path("./logs") / rel.with_suffix("")


def dump_config_snapshot(raw_config: Dict[str, Any], log_dir: Path) -> Path:
    """Dump raw yaml dict to log_dir for reproducibility."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dump_path = log_dir / f"config_dump_{timestamp}.yaml"
    with open(dump_path, "w", encoding="utf-8") as f:
        yaml.dump(raw_config, f, sort_keys=False, allow_unicode=True)
    logger.info(f"Config snapshot: {dump_path}")
    return dump_path


def _engine_config_to_dict(config: EngineConfig) -> Dict[str, Any]:
    resources = []
    for r in config.resources:
        item = {"id": r.id, "type": r.type}
        item.update(r.extra)
        resources.append(item)

    recipe = []
    for t in config.recipe:
        item = {
            "name": t.name,
            "type": t.type,
        }
        if t.model is not None:
            item["model"] = t.model
        if t.dataloader is not None:
            item["dataloader"] = t.dataloader
        if t.algorithm is not None:
            item["algorithm"] = t.algorithm
        if t.saver is not None:
            item["saver"] = t.saver
        item.update(t.extra)
        recipe.append(item)

    return {
        "metadata": {
            "name": config.metadata.name,
            "description": config.metadata.description,
        },
        "resources": resources,
        "recipe": recipe,
    }


def print_config(config: EngineConfig, title: str = "Configuration") -> None:
    """Pretty-print parsed v2 config via rich table and loguru."""
    cfg_dict = _engine_config_to_dict(config)
    console = Console(record=True, width=80)
    table = Table(show_header=False, show_lines=False, title=f"[bold blue]{title}[/bold blue]")
    table.add_column("Key", style="cyan", no_wrap=False)
    table.add_column("Value", style="magenta", overflow="fold")

    def add_items(d: Dict[str, Any], indent: int = 0) -> None:
        for key, value in d.items():
            prefix = "  " * indent + ("- " if indent > 0 else "")
            key_text = f"{prefix}{key}"
            if isinstance(value, dict):
                table.add_row(Text(key_text, style="bold green"), "")
                add_items(value, indent + 1)
            elif isinstance(value, list):
                table.add_row(Text(key_text, style="bold green"), f"[List with {len(value)} items]")
                for i, item in enumerate(value):
                    item_prefix = "  " * (indent + 1) + f"[{i}] "
                    if isinstance(item, dict):
                        type_name = item.get("type", "unknown")
                        table.add_row(Text(f"{item_prefix}type: {type_name}", style="yellow"), "")
                        add_items(item, indent + 2)
                    else:
                        table.add_row(item_prefix, str(item))
            else:
                table.add_row(key_text, str(value))

    for section, subconfig in cfg_dict.items():
        if not subconfig:
            continue
        table.add_row(Text(section.upper(), style="bold green"), "")
        if isinstance(subconfig, dict):
            add_items(subconfig, indent=1)
        elif isinstance(subconfig, list):
            add_items({section: subconfig}, indent=0)
        else:
            table.add_row(section, str(subconfig))
        table.add_section()

    console.print(table)
    logger.bind(quiet=True).info("\n" + console.export_text())
