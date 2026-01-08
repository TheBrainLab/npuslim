from typing import Optional
import argparse
import yaml
import time
from pathlib import Path
from loguru import logger
from easydict import EasyDict
from dataclasses import dataclass, field
from dacite import from_dict
from transformers.utils.hub import cached_file

from rich.console import Console
from rich.table import Table
from rich.text import Text


# ================================= Meta Config ================================= #
@dataclass(frozen=True)
class MetaConfig:
    type: str = field(metadata={"help": "Slim model type: 'llm' or 'vlm'"})
    save_path: str = field(metadata={"help": "Path to save the output model"})
    config_name: str = field(metadata={"help": "Configuration file name"})
    absolute_model_path: str = field(
        metadata={
            "help": "The absolute local file system path to the base model directory or checkpoint file."
        }
    )
    low_memory: bool = field(
        default=False, metadata={"help": "Enable low memory mode."}
    )


# ================================= Model Config ================================= #
@dataclass(frozen=True)
class ModelKwargs:
    """
    模型加载时传递给 Hugging Face from_pretrained 方法的额外关键字参数。
    """

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
        metadata={
            "help": "The dtype to use for the model parameters ('auto', 'float16', 'bfloat16', etc.)."
        },
    )
    device_map: str = field(
        default="cpu",
        metadata={
            "help": "Device to load the model on, typically 'cpu', 'auto', or a specific device ID."
        },
    )


@dataclass(frozen=True)
class TokenizerKwargs:
    """
    Tokenizer 或 Processor 加载时传递给 from_pretrained 方法的额外关键字参数。
    """

    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to allow custom code to be executed for the tokenizer/processor."
        },
    )
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "The specific model version (branch, commit ID, or tag) to use."
        },
    )


@dataclass(frozen=True)
class ModelConfig:
    type: str = field(metadata={"help": "Model identifier, e.g., 'Qwen3' or 'llama2'."})
    model_path: str = field(
        metadata={
            "help": "Local path to the model or remote identifier (e.g., Hugging Face ID)."
        }
    )
    model_kwargs: ModelKwargs = field(
        default_factory=ModelKwargs,
        metadata={
            "help": "Container for specific keyword arguments passed during model initialization."
        },
    )
    tokenizer_kwargs: TokenizerKwargs = field(
        default_factory=TokenizerKwargs,
        metadata={
            "help": "Container for keyword arguments passed during tokenizer/processor initialization."
        },
    )


# ================================= CalibDataset Config ================================= #
@dataclass(frozen=True)
class DatasetConfig:
    """
    数据集创建所需的配置。
    """

    type: str = field(
        metadata={"help": "Dataset Factory identifier, e.g., 'TextDataset'."}
    )
    data_path: str = field(metadata={"help": "The file path to the raw dataset."})
    num_samples: int = field(
        default=256,
        metadata={
            "help": "Maximum number of samples to use for calibration or processing."
        },
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum context length for tokenization and processing."},
    )
    device: str = field(
        default="cpu",
        metadata={
            "help": "The device to place processed data tensors on, e.g., 'cpu', 'npu:0'."
        },
    )


@dataclass(frozen=True)
class DataLoaderConfig:
    """
    PyTorch DataLoader 创建所需的配置。
    """

    batch_size: int = field(
        default=1, metadata={"help": "Batch size for the dataloader."}
    )
    shuffle: bool = field(
        default=True, metadata={"help": "Whether to shuffle the data."}
    )
    num_workers: int = field(
        default=0, metadata={"help": "Number of subprocesses to use for data loading."}
    )


@dataclass(frozen=True)
class CalibDatasetConfig:
    """
    校准数据集和其 DataLoader 的顶层配置。
    """

    dataset: DatasetConfig
    dataloader: DataLoaderConfig


# ================================= Compressor Config ================================= #
@dataclass(frozen=True)
class CompressorConfig: ...


@dataclass(frozen=True)
class PTQConfig(CompressorConfig):
    """
    PTQ配置.
    """

    type: str = field(
        metadata={
            "help": "The quantization algorithm type, e.g., 'INT8Dynamic', 'AWQ', 'GPTQ', etc."
        }
    )
    quant_config: EasyDict = field(
        metadata={
            "help": "Method-specific quantization parameters (e.g., w_bits, weight, activation)."
        },
    )
    ignore_layers: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names to skip during quantization (e.g., 'lm_head')."
        },
    )


@dataclass(frozen=True)
class SparseConfig(CompressorConfig):
    """
    Configuration for model sparsity (pruning).
    """

    type: str = field(
        metadata={
            "help": (
                "The sparsity algorithm type. Supported methods include: "
                "'SparseGPT' (Hessian-based pruning), 'Wanda' (Pruning by Weights and Activations), "
                "and 'Logit' (Logit-based importance)."
            )
        }
    )
    sparse_config: EasyDict = field(
        metadata={
            "help": (
                "Method-specific sparsity parameters:\n"
                "- sparsity_ratio: The target percentage of weights to be removed (e.g., 0.5 for 50%).\n"
                "- pattern: The sparsity structure. For Ascend NPU acceleration, '2:4' structured "
                "sparsity is mandatory (2 non-zero values out of every 4 consecutive elements)."
            )
        }
    )
    ignore_layers: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of module names to be skipped during the sparsity process to preserve model accuracy."
        }
    )


# ================================= 最顶层 Config ================================= #
@dataclass(frozen=True)
class FullConfig:
    """The complete top-level configuration structure."""

    meta: "MetaConfig"
    model: "ModelConfig"

    # Optional sections should be handled correctly
    calib_dataset: Optional["CalibDatasetConfig"] = field(default=None)
    ptq: Optional["PTQConfig"] = field(default=None)
    sparse: Optional["SparseConfig"] = field(default=None)


class SlimConfigParser:

    @staticmethod
    def from_args():
        args = SlimConfigParser.get_args()
        config = SlimConfigParser.load_config(args.config)
        config = SlimConfigParser.parser_config(args, config)
        SlimConfigParser.check_valid(config)
        SlimConfigParser.setup_logger(config.meta.save_path)
        SlimConfigParser.print_config(config, f"Configuration of {args.config}")
        return config

    @staticmethod
    def get_args():
        parser = argparse.ArgumentParser(description="NpuSlim")
        parser.add_argument("-c", "--config", type=str, required=True)
        parser.add_argument("--model-path", type=str, default=None)
        parser.add_argument("--save-path", type=str, default=None)
        parser.add_argument("--low-memory", action="store_true")
        # parser.add_argument("--multi-nodes", action="store_true")
        args = parser.parse_args()
        return args

    @staticmethod
    def load_config(path):
        with open(path, "r") as f:
            config = EasyDict(yaml.safe_load(f))
        return config

    @staticmethod
    def parser_config(args, config):
        def get_hf_model_path(model_path):
            p = Path(model_path)
            if p.is_file():
                return str(p)
            else:
                cached_config = Path(cached_file(model_path, "config.json"))
                return str(cached_config.parent)

        # config_name
        config.meta.config_name = Path(args.config).stem
        # model_path
        if args.model_path is not None:
            config.model.model_path = args.model_path
        # absolute_model_path
        config.meta.absolute_model_path = get_hf_model_path(config.model.model_path)
        # low_memory
        if args.low_memory:
            config.meta.low_memory = True
        # save_path
        if args.save_path is not None:
            config.meta.save_path = args.save_path
        else:
            save_path = Path(config.meta.save_path) / config.meta.config_name
            config.meta.save_path = str(save_path)
        return config

    @staticmethod
    def check_valid(config):
        slim_type = config.meta.type.lower()
        assert slim_type in [
            "llm",
            "vlm",
        ], f"Unsupported slim model type: {config.meta.type}. Must be 'llm' or 'vlm'."
        assert (
            "model" in config and config.model
        ), "Missing 'model' configuration in YAML."
    
    @staticmethod
    def setup_logger(save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}.log"
        log_file = save_path / log_filename
        
        logger.add(str(log_file), rotation="500 MB", encoding="utf-8", enqueue=True)
        logger.info(f"Log will be saved to: {log_file}")

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
                    value_str = "\n".join(str(i) for i in v)
                    table.add_row(key_str, value_str)
                else:
                    table.add_row(key_str, str(v))

        for section, subconfig in config.items():
            table.add_row(Text(section.upper(), style="bold green"), "")
            add_items(subconfig, indent=1)
            table.add_section()

        console.print(table)


class GlobalConfig:
    _instance = None
    _cfg = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GlobalConfig, cls).__new__(cls)
            cls._cfg = cls._load_config()
        return cls._instance

    @staticmethod
    def _load_config():
        try:
            cfg = SlimConfigParser.from_args()
            cfg_dict = dict(cfg)
            final_cfg = from_dict(
                data_class=FullConfig,
                data=cfg_dict,
            )
            return final_cfg

        except Exception as e:
            raise RuntimeError(
                f"Failed to load or validate configuration via SlimConfigParser: {e}"
            )

    @classmethod
    def get_config(cls):
        if cls._cfg is None:
            cls()
        return cls._cfg
