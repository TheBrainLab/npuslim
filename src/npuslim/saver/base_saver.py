from abc import ABC, abstractmethod
from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path
from loguru import logger

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel


class BaseSaver(ABC):
    """
    Abstract Base Class for saving strategies (Strategy Pattern).
    Provides a template for persisting different model formats.
    """

    def __init__(self, model: "BaseLLMModel", config: Dict[str, Any]):
        """
        Initializes the saver with a model wrapper and format-specific configuration.
        """
        self.model = model
        self.config = config

    def save(self, save_path: str | Path):
        """
        Public unified interface for saving the model and its components.
        Handles directory creation and logging.
        """
        save_path = Path(save_path)
        try:
            # Ensure the target directory exists before saving
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Saving to {save_path}...")

            # Execute the specific weight-saving logic implemented by subclasses
            self._save_impl(save_path)

            logger.success(f"✅ Save completed: {save_path}")
        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            raise e

    @abstractmethod
    def _save_impl(self, save_path: Path):
        """
        Internal implementation for saving model weights. 
        Must be implemented by subclasses (e.g., HuggingFaceSaver, GPTQSaver).
        """
        pass

    def save_tokenizer(self, save_path: Path):
        """
        Common utility to save the tokenizer configuration and vocabulary files.
        """
        # Ensure the model wrapper provides a tokenizer attribute
        if getattr(self.model, "tokenizer", None):
            try:
                self.model.tokenizer.save_pretrained(save_path)
                logger.debug("Tokenizer config saved.")
            except Exception as e:
                logger.warning(f"Failed to save tokenizer: {e}")
        else:
            logger.debug("No tokenizer found to save.")

    def save_processor(self, save_path: Path):
        """
        Common utility to save the processor (primarily for VLMs or multi-modal models).
        """
        # Some LLMs use a separate processor, while others use the tokenizer directly
        processor = getattr(self.model, "processor", None)

        if processor:
            try:
                # Save processor files (e.g., preprocessor_config.json) to the target path
                processor.save_pretrained(save_path)
                logger.debug("Processor config saved.")
            except Exception as e:
                logger.warning(f"Failed to save processor: {e}")
        else:
            logger.debug("No processor found to save.")