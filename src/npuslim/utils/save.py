from typing import Any, TYPE_CHECKING
import yaml
from pathlib import Path
from loguru import logger
from dataclasses import asdict, is_dataclass

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel


class SlimSaver:
    def __init__(self, save_path: str | Path, config: Any):
        self.save_path = Path(save_path)
        self.config = config

    def save(self, model: "BaseLLMModel", pipeline: list):
        self.save_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting saving process at {self.save_path}")

        self._save_model(model)
        self._save_task_metadata(pipeline)
        self._save_config()

        logger.success("Saving completed.")

    def _save_model(self, model: "BaseLLMModel"):
        logger.info(f"Saving model to {self.save_path}...")
        model.save_pretrained(self.save_path)
        logger.success(f"Successfully saved model to {self.save_path}.")

    def _save_task_metadata(self, pipeline: list):
        for task in pipeline:
            if hasattr(task, "save"):
                task_name = task.__class__.__name__
                task.save(self.save_path)
                logger.success(f"Metadata for {task_name} saved.")

    def _save_config(self):
        def to_standard_dict(obj):
            if is_dataclass(obj):
                return {k: to_standard_dict(v) for k, v in asdict(obj).items()}
            elif isinstance(obj, dict):
                return {k: to_standard_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_standard_dict(v) for v in obj]
            else:
                return obj
            
        config_path = self.save_path / "slim_config.yaml"
        config_dict = to_standard_dict(self.config)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, sort_keys=False)
        logger.success(f"Config saved to {config_path}")
