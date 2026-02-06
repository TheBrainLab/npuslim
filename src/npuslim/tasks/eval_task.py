import torch
import json
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from pathlib import Path
from loguru import logger

from npuslim.utils.factory import TaskFactory
from npuslim.utils.backend import bh
from .base_task import BaseTask


__all__ = ["EvalTask"]


@dataclass
class EvalTaskConfig:
    """
    Configuration for EvalTask.
    Minimalist version focused on in-memory accuracy validation.
    """

    # --- General Settings ---
    output_dir: str = "outputs"

    # --- Accuracy Eval (lmeval) Settings ---
    datasets: List[str] = field(default_factory=lambda: ["wikitext"])
    num_fewshot: int = 0
    batch_size: Union[int, str] = 1
    limit: Optional[int] = None


@TaskFactory.register("eval")
class EvalTask(BaseTask):
    """
    Task to evaluate model quality immediately after compression.
    This provides a quick sanity check before the model is used in external benchmarks.
    """

    ConfigClass = EvalTaskConfig

    def execute(self):
        """
        Main execution flow: Cleanup -> Path Resolution -> Hardware Migration -> Backend Routing.
        """
        # 1. Clean up stale memory to prevent OOM during evaluation
        logger.info(f"🧹 [EvalTask] Vacuuming memory on {bh.name}...")
        bh.full_vacuum()

        # 2. Extract model and tokenizer from shared resources
        torch_model = self.model.model
        tokenizer = self.model.tokenizer

        # 3. Resolve the path where the sanity check results will be logged
        final_output_dir = self._resolve_output_dir()
        logger.info(
            f"📂 [EvalTask] Sanity check results will be saved to: {final_output_dir}"
        )

        # 4. Ensure model is on the current compute backend (NPU/GPU)
        self._ensure_model_on_backend(torch_model)

        # 5. Run the in-memory accuracy evaluation
        self._run_accuracy_eval(torch_model, tokenizer)

        # 6. Final memory synchronization and cleanup
        bh.empty_cache()

    def _resolve_output_dir(self) -> Path:
        """
        Resolves the final output directory by mirroring the config file structure.
        """
        meta = self.engine.cfg.meta
        if meta.config_path:
            rel_path = Path(meta.config_path).with_suffix("")
        else:
            rel_path = Path("unknown_config")

        base_dir = Path(self.cfg.output_dir)
        # Results are placed under 'eval' subfolder to keep 'outputs' clean
        return (base_dir / "eval" / rel_path).resolve()

    def _ensure_model_on_backend(self, model: torch.nn.Module):
        """
        Transfers the model to the target hardware device if it's currently on CPU.
        """
        current_device = next(model.parameters()).device
        if current_device.type != bh.name:
            try:
                logger.info(
                    f"🚚 [EvalTask] Moving model: {current_device} -> {bh.device}"
                )
                model.to(bh.device)
                bh.sync()
            except Exception as e:
                logger.warning(
                    f"⚠️ [EvalTask] Migration failed: {e}. Proceeding with current device."
                )
        else:
            logger.info(f"✅ [EvalTask] Model is on target device: {current_device}")

    # ========================== Evaluation Backend ========================== #

    def _run_accuracy_eval(self, model, tokenizer):
        """
        Executes accuracy evaluation using lm-evaluation-harness (HFLM backend).
        """
        try:
            import lm_eval
            from lm_eval.utils import make_table
            from lm_eval.models.huggingface import HFLM
        except ImportError:
            raise ImportError("Please install lm-eval: pip install lm-eval")

        logger.info(f"📊 [EvalTask] Starting Accuracy Evaluation...")

        # 1. Wrap the in-memory torch model for lmeval
        lm_obj = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=self.cfg.batch_size,
            device=bh.device,
        )

        # 2. Automatically detect and apply chat templates if available
        use_chat_template = False
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            use_chat_template = True
            logger.info(
                "ℹ️ [EvalTask] Chat template detected. Enabling 'apply_chat_template'."
            )

        # 3. Perform evaluation
        results = lm_eval.simple_evaluate(
            model=lm_obj,
            tasks=self.cfg.datasets,
            num_fewshot=self.cfg.num_fewshot,
            limit=self.cfg.limit,
            log_samples=True,  # Captures per-sample info for debugging
            write_out=False,  # Suppresses excessive stdout logs
            apply_chat_template=use_chat_template,
        )

        # 4. Log results table to console
        if results:
            table_str = make_table(results)
            logger.info(f"\n📈 [Sanity Check Results]:\n{table_str}")

        # 5. Persist detailed results to disk
        if self.cfg.output_dir and results:
            self._save_results(results)

        logger.success("✨ Accuracy evaluation completed.")

    def _save_results(self, results: dict):
        """
        Saves the results dictionary to a JSON file with type handling for NumPy.
        """
        try:
            output_path = self._resolve_output_dir()
            output_path.mkdir(parents=True, exist_ok=True)

            # Construct a descriptive filename based on task names
            task_str = "_".join(self.cfg.datasets)[:50]
            filename = f"{task_str}_{self.cfg.num_fewshot}shot.json"
            file_path = output_path / filename

            def default_serializer(obj: Any) -> Any:
                """Ensures NumPy types and sets are JSON serializable."""
                if isinstance(obj, set):
                    return list(obj)
                if hasattr(obj, "dtype") and hasattr(obj, "item"):
                    return obj.item()
                return str(obj)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=default_serializer)

            logger.info(f"💾 [EvalTask] Results saved to: {file_path}")

        except Exception as e:
            logger.error(f"❌ [EvalTask] Failed to save results JSON: {e}")
