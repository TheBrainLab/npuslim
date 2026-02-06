from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger

from npuslim.tasks.base_task import BaseTask
from npuslim.utils.factory import TaskFactory, SaverFactory


__all__ = ["SaveTask"]


@dataclass
class SaveTaskConfig:
    """
    Configuration for model saving operations.
    Determines where and in what format the processed model will be stored.
    """
    type: str = "save"
    algo_name: str = "SaveHFModel"     # The saver algorithm to use (mapped in saver/__init__.py)
    save_path: Optional[str] = None   # Absolute path (highest priority)
    save_dir: Optional[str] = None    # Base directory to join with config filename
    fallback_dirname: str = "checkpoints" # Default folder name if no path is provided
    saver_args: Dict[str, Any] = field(default_factory=dict) # Specific arguments for the saver


@TaskFactory.register("save")
class SaveTask(BaseTask):
    """
    Task responsible for persisting the model to disk.
    It acts as a wrapper around different Saver implementations.
    """
    ConfigClass = SaveTaskConfig

    def execute(self):
        # 1. Determine the final physical location on disk
        final_save_path = self._resolve_save_path()

        logger.info(f"💾 [SaveTask] Saving model...")
        logger.info(f"   -> Format: {self.cfg.algo_name}")
        logger.info(f"   -> Target: {final_save_path}")

        # Ensure the target directory exists
        final_save_path.mkdir(parents=True, exist_ok=True)

        if self.model is None:
            logger.error("❌ Main model is not initialized. Cannot save.")
            return

        try:
            # 2. Instantiate the specific saver through SaverFactory (e.g., HuggingFaceSaver)
            # This triggers a lazy import of the specific saver module.
            saver = SaverFactory.create(
                format_name=self.cfg.algo_name,
                model=self.model,
                config=self.cfg.saver_args,
            )
            
            # 3. Perform the actual save operation
            saver.save(final_save_path)
            
            # Update the model's metadata with the new location if applicable
            if hasattr(self.model, "model_path"):
                self.model.model_path = str(final_save_path)

            logger.success(f"✨ [SaveTask] Completed. Saved to: {final_save_path}")

        except Exception as e:
            logger.error(f"❌ Save failed: {e}")
            raise e

    def _resolve_save_path(self) -> Path:
        """
        Logic to decide the final saving path based on priority:
        1. Explicit save_path
        2. save_dir + original config filename
        3. Global work_dir + fallback_dirname
        """
        meta = self.engine.cfg.meta
        # Get the name of the config file used for the run to create distinct subfolders
        rel_path = (
            Path(meta.config_path) if meta.config_path else Path("unknown_config")
        )
        rel_path = rel_path.with_suffix("")

        # Priority 1: User explicitly provided a full path
        if self.cfg.save_path:
            return Path(self.cfg.save_path).resolve()

        # Priority 2: User provided a base directory, append the config filename
        if self.cfg.save_dir:
            return (Path(self.cfg.save_dir) / rel_path).resolve()

        # Priority 3: Fallback to the global workspace directory defined in Engine
        base_work_dir = Path(meta.work_dir)
        return (base_work_dir / self.cfg.fallback_dirname).resolve()