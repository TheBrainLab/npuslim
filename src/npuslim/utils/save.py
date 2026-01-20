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

        self._save_model(model, pipeline)
        self._save_task_metadata(pipeline)
        self._save_config()

        logger.success("Saving process finished.")

    def _save_model(self, model: "BaseLLMModel", pipeline: list):
        """
        询问最后一个任务是否要接管模型保存，避免中间过程重复写入。
        """
        is_handled = False
        
        for task in pipeline:
            if hasattr(task, "save_model"):
                if task.save_model(self.save_path):
                    logger.info(f"Model saving handled by task: {task.__class__.__name__}")
                    is_handled = True
        
        if not is_handled:
            logger.info("Executing default model saving (save_pretrained)...")
            model.save_pretrained(self.save_path)

        logger.success(f"Model saved at {self.save_path}")

    def _save_task_metadata(self, pipeline: list):
        """
        元数据依然建议遍历，因为稀疏任务和量化任务通常需要各自生成不同的 JSON 描述。
        """
        for task in pipeline:
            if hasattr(task, "save_meta"):
                task.save_meta(self.save_path)
                logger.success(f"Metadata for {task.__class__.__name__} saved.")

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