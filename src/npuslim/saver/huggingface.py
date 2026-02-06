from pathlib import Path
from loguru import logger
from .base_saver import BaseSaver
from npuslim.utils.factory import SaverFactory


__all__ = ["HuggingFaceSaver"]


@SaverFactory.register("HuggingFaceSaver")
class HuggingFaceSaver(BaseSaver):
    """
    Implementation for saving models in the standard Hugging Face format.
    """

    def _save_impl(self, save_path: Path):
        # 1. Extract saving configurations from the config dictionary
        safe_serialization = self.config.get("safe_serialization", True)
        max_shard_size = self.config.get("max_shard_size", "5GB")

        logger.info(f"   -> Mode: {'SafeTensors' if safe_serialization else 'Bin'}")

        # 2. Persist model weights using the underlying transformers API
        self.model.model.save_pretrained(
            save_path,
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
        )

        # 3. Persist auxiliary components using common base class utilities
        self.save_tokenizer(save_path)
        self.save_processor(save_path)