from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


class BaseLLMModel(ABC):
    def __init__(self, *args, model_type, model_path, trust_remote_code=False, **kwargs):
        self.model_type = model_type
        self.model_path = model_path
        self.trust_remote_code = trust_remote_code

        self.model_kwargs = kwargs.pop("model_kwargs", kwargs)
        self.tokenizer_kwargs = kwargs.pop("tokenizer_kwargs", {})

        self.model = None
        self.tokenizer = None

    def prepare(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            **self.model_kwargs, 
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            trust_remote_code=self.trust_remote_code,
            **self.tokenizer_kwargs,  
        )
    
    def init_ptq(self, slim_config):
        """
        Initialize the model for post-training quantization (PTQ).
        Args:
            slim_config(dict, required): the configuration for quantization.
                - compress_config: the configuration for compression.
                - global_config: the global configuration for the model.
        """
        # quant_config = QuantConfig(
        #     slim_config["compress_config"], slim_config["global_config"]
        # )
        # self.quant_config = quant_config
        self.act_scales_dict = {}
        self.weight_scales_dict = {}
        self.weight_scales_dict_2 = {}
        self.kv_cache_scales_dict = {}
        if hasattr(self.quant_config, "weight_observer"):
            self.quant_algo_dict = self.get_quant_config()
        else:
            self.quant_algo_dict = None
        self.quantized = False


