"""NPUSlim CLI entry point."""
import argparse
from pathlib import Path

from npuslim import SlimEngine


def parse_args():
    parser = argparse.ArgumentParser(description="NPUSlim: LLM Quantization Framework")
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to config file (YAML)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config.strip())
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    engine = SlimEngine(cfg_path)
    engine.run()


if __name__ == "__main__":
    main()