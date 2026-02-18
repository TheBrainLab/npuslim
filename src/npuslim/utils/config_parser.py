from typing import Optional, List, Dict, Any
import argparse
import yaml
import time
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from dacite import from_dict, Config as DaciteConfig

from rich.console import Console
from rich.table import Table
from rich.text import Text


# ================================= Meta Config ================================= #
@dataclass(frozen=True)
class MetaConfig:
    type: str = field(metadata={"help": "Slim model type: 'llm' or 'vlm'"})
    config_path: str = field(
        metadata={
            "help": "Relative configuration path identifier, e.g., 'compressor/gptq/qwen3_8bit.yaml'"
        }
    )
    low_memory: bool = field(
        default=False, metadata={"help": "Enable low memory mode."}
    )
    work_dir: str = field(
        default="./work_dirs",
        metadata={"help": "Directory for logs, config dumps, and runtime artifacts."},
    )


# ================================= Model Config ================================= #
@dataclass(frozen=True)
class ModelKwargs:
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow custom code to be executed for this model."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether to use the 'low_cpu_mem_usage' optimization."},
    )
    use_cache: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the model should return the last key/value cache."
        },
    )
    torch_dtype: str = field(
        default="auto",
        metadata={"help": "The dtype to use for the model parameters."},
    )
    device_map: str = field(
        default="cpu",
        metadata={"help": "Device to load the model on."},
    )


@dataclass(frozen=True)
class TokenizerKwargs:
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether to allow custom code to be executed."},
    )
    revision: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model version."},
    )


@dataclass(frozen=True)
class ModelConfig:
    type: str = field(metadata={"help": "Model identifier, e.g., 'Qwen3'."})
    model_path: str = field(metadata={"help": "Local path or Hugging Face ID."})
    model_hub: str = field(
        default="hf",
        metadata={"help": "Model hub source, e.g., 'ms' or 'hf'."},
    )
    model_kwargs: ModelKwargs = field(
        default_factory=ModelKwargs,
        metadata={"help": "Specific keyword arguments for model initialization."},
    )
    tokenizer_kwargs: TokenizerKwargs = field(
        default_factory=TokenizerKwargs,
        metadata={"help": "Keyword arguments for tokenizer initialization."},
    )


# ================================= CalibDataset Config ================================= #
@dataclass(frozen=True)
class DatasetConfig:
    type: str = field(metadata={"help": "Dataset Factory identifier."})
    data_path: Optional[str] = field(default=None, metadata={"help": "Path to the raw dataset (optional for some datasets like C4)."})
    num_samples: int = field(default=256, metadata={"help": "Max samples."})
    max_seq_length: int = field(default=2048, metadata={"help": "Max context length."})
    device: str = field(default="cpu", metadata={"help": "Device for data tensors."})
    seed: int = field(default=0, metadata={"help": "Random seed for sampling."})


@dataclass(frozen=True)
class DataLoaderConfig:
    batch_size: int = field(default=1, metadata={"help": "Batch size."})
    shuffle: bool = field(default=True, metadata={"help": "Shuffle data."})
    num_workers: int = field(default=0, metadata={"help": "Subprocesses for loading."})


@dataclass(frozen=True)
class CalibDatasetConfig:
    dataset: DatasetConfig
    dataloader: DataLoaderConfig


# ================================= Full Config ================================= #
@dataclass(frozen=True)
class FullConfig:
    meta: MetaConfig
    model: ModelConfig
    calib_dataset: Optional[CalibDatasetConfig] = field(default=None)
    speculative: Optional[Dict[str, Any]] = field(default=None)
    distillation: Optional[Dict[str, Any]] = field(default=None)
    pipeline: List[Dict[str, Any]] = field(default_factory=list)


class SlimConfigParser:

    @staticmethod
    def from_args():
        args = SlimConfigParser.get_args()
        config_dict = SlimConfigParser.load_config(args.config)
        config_dict = SlimConfigParser.merge_args_into_config(args, config_dict)

        meta = config_dict.get("meta", {})
        work_dir = meta.get("work_dir")
        SlimConfigParser.setup_logger(work_dir)
        SlimConfigParser.dump_config(config_dict, work_dir)
        SlimConfigParser.print_config(config_dict, f"Configuration of {args.config}")

        return config_dict

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="NpuSlim")
        parser.add_argument("-c", "--config", type=str, required=True)
        parser.add_argument("--model-path", type=str, default=None)
        parser.add_argument(
            "--work-dir",
            type=str,
            default=None,
            help="Directory for logs and config snapshots.",
        )

        parser.add_argument("--low-memory", action="store_true")
        return parser.parse_args()

    @staticmethod
    def load_config(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def merge_args_into_config(args, config):
        if "meta" not in config:
            config["meta"] = {}

        config_p = Path(args.config).resolve()
        parts = config_p.parts
        start_idx = -1
        for i in range(len(parts) - 2, -1, -1):
            if parts[i] in ["config", "configs"]:
                start_idx = i + 1
                break

        if start_idx != -1:
            rel_path = Path(*parts[start_idx:])
        else:
            try:
                rel_path = config_p.relative_to(Path.cwd())
            except ValueError:
                rel_path = Path(config_p.name)

        config["meta"]["config_path"] = str(rel_path)
        if args.model_path is not None:
            if "model" not in config:
                config["model"] = {}
            config["model"]["model_path"] = args.model_path

        if args.low_memory:
            config["meta"]["low_memory"] = True

        yaml_work_dir = config["meta"].get("work_dir")
        if args.work_dir is not None:
            final_work_dir = args.work_dir
        elif yaml_work_dir is not None:
            final_work_dir = yaml_work_dir
        else:
            final_work_dir = "./work_dirs"

        rel_path_no_ext = rel_path.with_suffix("")
        final_work_dir = Path(final_work_dir) / rel_path_no_ext
        config["meta"]["work_dir"] = str(final_work_dir)

        return config

    @staticmethod
    def dump_config(config_dict, work_dir):
        try:
            if not work_dir:
                return
            path = Path(work_dir)
            path.mkdir(parents=True, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dump_path = path / f"config_dump_{timestamp}.yaml"

            with open(dump_path, "w", encoding="utf-8") as f:
                yaml.dump(config_dict, f, sort_keys=False, allow_unicode=True)
            logger.info(f"💾 Config snapshot saved to: {dump_path}")
        except Exception as e:
            logger.warning(f"Failed to dump config backup: {e}")

    @staticmethod
    def check_valid(config: FullConfig):
        slim_type = config.meta.type.lower()
        assert slim_type in [
            "llm",
            "vlm",
        ], f"Unsupported slim model type: {config.meta.type}. Must be 'llm' or 'vlm'."

        assert config.model is not None, "Missing 'model' configuration."
        if not config.pipeline and not config.speculative and not config.distillation:
            logger.warning(
                "Pipeline is empty and no speculative/distillation config found."
            )

    @staticmethod
    def setup_logger(save_path):
        import sys

        def format_filename(record):
            file_name = record["file"].name
            max_len = 15
            if len(file_name) > max_len:
                file_name = file_name[: (max_len - 2)] + ".."

            record["extra"]["short_file"] = f"{file_name: <{max_len}}"

        logger.remove()
        logger.configure(patcher=format_filename)
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[short_file]}:{line: <4}</cyan> | "
            "<level>{message}</level>"
        )

        logger.add(
            sys.stderr,
            format=log_format,
            level="INFO",
        )

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_filename = f"{timestamp}.log"
            log_file = save_path / log_filename

            logger.add(
                str(log_file),
                format=log_format,
                rotation="500 MB",
                encoding="utf-8",
                enqueue=True,
            )

        logger.info("=" * 60)
        logger.info("🚀 NPUSlim Quantization Framework Initialized")
        logger.info(f"📂 Log Directory : {save_path}")
        logger.info(f"📄 Current Log   : {log_filename}")
        logger.info("=" * 60)

    @staticmethod
    def print_config(config: dict, title: str = "Configuration"):
        console = Console()
        table = Table(
            show_header=False, show_lines=False, title=f"[bold blue]{title}[/bold blue]"
        )
        table.add_column("Key", style="cyan", no_wrap=False)
        table.add_column("Value", style="magenta", overflow="fold")

        def add_items(d, indent=0):
            for k, v in d.items():
                prefix = "  " * indent + ("- " if indent > 0 else "")
                key_str = f"{prefix}{k}"

                if isinstance(v, dict):
                    table.add_row(Text(key_str, style="bold green"), "")
                    add_items(v, indent=indent + 1)
                elif isinstance(v, list):
                    table.add_row(
                        Text(key_str, style="bold green"), f"[List with {len(v)} items]"
                    )
                    for i, item in enumerate(v):
                        item_prefix = "  " * (indent + 1) + f"[{i}] "
                        if isinstance(item, dict):
                            item_name = item.get("type", "unknown")
                            table.add_row(
                                Text(f"{item_prefix}type: {item_name}", style="yellow"),
                                "",
                            )
                            add_items(item, indent=indent + 2)
                        else:
                            table.add_row(f"{item_prefix}", str(item))
                else:
                    table.add_row(key_str, str(v))

        for section, subconfig in config.items():
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


class GlobalConfig:
    _instance = None
    _cfg: Optional[FullConfig] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._cfg = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config():
        cfg_dict = SlimConfigParser.from_args()
        try:
            final_cfg = from_dict(
                data_class=FullConfig, data=cfg_dict, config=DaciteConfig(strict=False)
            )
        except Exception as e:
            logger.error(f"Configuration parsing failed: {e}")
            raise e

        SlimConfigParser.check_valid(final_cfg)
        return final_cfg

    @classmethod
    def get_config(cls) -> FullConfig:
        if cls._cfg is None:
            cls()
        return cls._cfg
