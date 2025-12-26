from abc import ABC, abstractmethod
import torch
from dataclasses import asdict
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from npuslim.utils.config_parser import ModelConfig


class BaseLLMModel(ABC):
    def __init__(self, *args, config: "ModelConfig", **kwargs):
        self.model_path = config.model_path
        self.model_kwargs = config.model_kwargs
        self.tokenizer_kwargs = config.tokenizer_kwargs

        self.model = None
        self.tokenizer = None
        self.config = None

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

    def init_ptq(self):
        """
        Initialize the model for post-training quantization (PTQ).
        """
        self.act_scales_dict = {}
        self.weight_scales_dict = {}
        self.kv_cache_scales_dict = {}
        self.quantized = False
    
    def get_weight_scales(self, layer, weight_observer):
        weight = layer.weight.clone().detach()
        weight_observer(weight)
        return weight_observer.scales()
    
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
    def get_observer_layers(self): ...

    @abstractmethod
    def get_save_func(self): ...
