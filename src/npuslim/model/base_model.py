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


