from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import torch
from dataclasses import asdict
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from npuslim.utils.config_parser import GlobalConfig

if TYPE_CHECKING:
    from npuslim.utils.config_parser import ModelConfig


class BaseLLMModel(ABC):
    def __init__(self, *args, config: "ModelConfig", **kwargs):
        self.model_path = config.model_path
        self.model_kwargs = config.model_kwargs
        self.tokenizer_kwargs = config.tokenizer_kwargs
        self.low_memory = GlobalConfig.get_config().meta.low_memory

        self.model = None
        self.tokenizer = None
        self.config = None
        self.quantized = False

        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.observer_layer_classes = [torch.nn.Linear]

    def prepare(self):
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

        logger.info(f"Loading configuration from: '{self.model_path}'")
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
        )
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
