import json
from pathlib import Path
from loguru import logger

from .quant_algo_info import QuantConfigManager
from npuslim.model.base_model import BaseLLMModel


class Saver:
    def __init__(self, save_model: BaseLLMModel, save_path: str):
        self.save_model = save_model
        self.save_path = Path(save_path).resolve()
        self.quant_info = QuantConfigManager.get_config()
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created export directory: {self.save_path}")

    def save_quant_config(self):
        quant_algo = self.quant_info.quant_algo
        quant_layer_names = self.quant_info.observer_layers_names
        quant_model_description = self.quant_info.quant_model_description
        save_path = self.save_path / "quant_model_description.json"

        for key in self.save_model.model.state_dict().keys():
            matched_layer = None
            for q_layer in quant_layer_names:
                if key.startswith(q_layer + "."):
                    matched_layer = q_layer
                    break

            if matched_layer:
                quant_model_description[key] = quant_algo
            else:
                quant_model_description[key] = "FLOAT"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(quant_model_description, f, indent=4, ensure_ascii=False)

    def save(self):
        self.save_quant_config()
        self.save_model.model.save_pretrained(self.save_path)
        components = ["model weights"]
        if hasattr(self.save_model, "tokenizer"):
            self.save_model.tokenizer.save_pretrained(self.save_path)
            components.append("tokenizer")
        logger.info(f"Successfully saved {', '.join(components)} to: {self.save_path}")
