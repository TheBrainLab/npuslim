"""NPUSlim CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from npuslim import SlimEngine
from npuslim.core.bootstrap import bootstrap_from_path


def parse_args():
    parser = argparse.ArgumentParser(description="NPUSlim: LLM Quantization Framework")
    parser.add_argument(
        "config",
        nargs="?",
        type=str,
        help="Path to config file (YAML)",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        dest="config_flag",
        help="Path to config file (YAML)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for logs and config snapshots",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Disable NPUSlim ASCII header output",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config_str = (args.config_flag or args.config or "").strip()
    if not config_str:
        raise SystemExit("Error: config path is required (positional or -c/--config)")
    cfg_path = Path(config_str)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    parsed_cfg = bootstrap_from_path(
        cfg_path,
        log_dir=args.log_dir,
        show_header=not args.no_header,
        strict_validate=True,
    )

    engine = SlimEngine(parsed_cfg)
    engine.run()


if __name__ == "__main__":
    main()
