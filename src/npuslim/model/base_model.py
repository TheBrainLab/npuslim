from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch
import importlib
from dataclasses import asdict
from loguru import logger
from npuslim.utils.config_parser import GlobalConfig

if TYPE_CHECKING:
    from npuslim.utils.config_parser import ModelConfig


def get_hub_class(model_hub: str, class_name: str):
    """
    Dynamically fetch the specified class from the given hub (hf/ms).
    
    :param model_hub: 'hf' (Hugging Face) or 'ms' (ModelScope)
    :param class_name: The name of the class to import, e.g., 'AutoModelForCausalLM'
    :return: The class object
    """
    hub_pkg_map = {
        "hf": "transformers",
        "ms": "modelscope"
    }

    if model_hub not in hub_pkg_map:
        raise ValueError(f"❌ Unsupported hub: {model_hub}. Supported hubs are 'hf' and 'ms'.")
    pkg_name = hub_pkg_map[model_hub]

    try:
        module = importlib.import_module(pkg_name)
        cls = getattr(module, class_name)
        return cls

    except ImportError:
        raise ImportError(f"⚠️ '{pkg_name}' not installed. Please run: pip install {pkg_name}")
    except AttributeError:
        raise AttributeError(
            f"⚠️ Class '{class_name}' not found in '{pkg_name}'. "
            f"Please check the class name spelling or package version."
        )


class BaseLLMModel(ABC):
    def __init__(self, *args, config: "ModelConfig", **kwargs):
        self.model_path = config.model_path
        self.model_hub = config.model_hub
        self.model_kwargs = config.model_kwargs
        self.tokenizer_kwargs = config.tokenizer_kwargs
        self.low_memory = GlobalConfig.get_config().meta.low_memory
        self.skip_layer_names = ["lm_head"]

        self.model = None
        self.tokenizer = None
        self.config = None
        self.quantized = False
        self.model_type = "LLM"

        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.observer_layer_classes = [torch.nn.Linear]

    def prepare(self):
        AutoModelForCausalLM = get_hub_class(self.model_hub, "AutoModelForCausalLM")
        AutoTokenizer = get_hub_class(self.model_hub, "AutoTokenizer")
        
        logger.info(
            f"Loading model from: '{self.model_path}' with kwargs: {self.model_kwargs}"
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path, **asdict(self.model_kwargs)
        )

        logger.info(
            f"Loading tokenizer from: '{self.model_path}' with kwargs: {self.tokenizer_kwargs}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            **asdict(self.tokenizer_kwargs),
        )
        self.config = self.model.config

        logger.success(
            f"Model, tokenizer, and config loaded successfully. "
            f"Model architecture: {self.config.architectures[0] if hasattr(self.config, 'architectures') and self.config.architectures else 'N/A'}"
        )

    def forward(self, *args, **kwargs):
        if not self.low_memory:
            return self.model(*args, **kwargs)
        else:
            # TODO
            raise NotImplementedError
            # return self._streaming_forward(*args, **kwargs)

    def get_pre_transformer_modules(self):
        pre_transformer_modules_dict = {}
        for full_name in self.pre_transformer_module_names:
            current_module = self.model
            parts = full_name.split(".")
            for part in parts:
                if not hasattr(current_module, part):
                    current_module = None
                    break
                current_module = getattr(current_module, part)
            if current_module is not None:
                pre_transformer_modules_dict[full_name] = current_module
        return pre_transformer_modules_dict

    def get_kvcache_observer_layers_names(self, observe_names):
        names = ["self_attn.k_proj", "self_attn.v_proj"]
        return [
            k
            for k in observe_names
            if k.startswith(self.block_name)
            and k.split(".")[-2] + "." + k.split(".")[-1] in names
        ]

    def get_quant_convert_module(self):
        """
        Returns the module that will be converted to quantized.
        This is typically the main transformer module of the model.
        """
        return self.model

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path, safe_serialization=True)
        self.tokenizer.save_pretrained(save_path)

    @abstractmethod
    def get_observer_layers(self, ignore_layers: list = []): ...

    def get_layers(self): ...

    # @abstractmethod
    # def get_save_func(self): ...
