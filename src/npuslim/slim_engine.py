from dataclasses import asdict
from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from torch.utils.data import DataLoader

from npuslim.utils.config_parser import GlobalConfig
from npuslim.utils.save import SlimSaver
from npuslim.utils.factory import ModelFactory, DatasetFactory
from npuslim.tasks import SparseTask, PTQTask, SpeculativeTask

if TYPE_CHECKING:
    from npuslim.model.base_model import BaseLLMModel
    from npuslim.tasks.base_task import BaseTask


class SlimEngine:
    def __init__(self):
        self.cfg = GlobalConfig.get_config()

        self.model: "BaseLLMModel" = None
        self.dataloader: Optional[DataLoader] = None
        self.pipeline: List["BaseTask"] = []
        self.saver = SlimSaver(save_path=self.cfg.meta.save_path, config=self.cfg)

        self.prepare_resources()
        self.build_pipeline()

    def prepare_resources(self):
        logger.info("Preparing resources...")

        self.model = self._init_model(self.cfg.model)

        if hasattr(self.cfg, "speculative") and self.cfg.speculative:
            # TODO
            logger.info("Speculative decoding config detected. Loading draft model...")
            self.draft_model = self._init_model(self.cfg.speculative.draft_model)

        self.prepare_dataloader()

    def _init_model(self, model_cfg):
        model = ModelFactory.create(config=model_cfg)
        model.prepare()
        return model

    def prepare_dataloader(self):
        if not hasattr(self.cfg, "calib_dataset") or self.cfg.calib_dataset is None:
            logger.warning("No calibration dataset config found.")
            return

        processor = (
            self.model.processor
            if self.cfg.meta.type.lower() == "vlm"
            else self.model.tokenizer
        )
        dataset = DatasetFactory.create(
            processor=processor, config=self.cfg.calib_dataset.dataset
        )
        dataloader_kwargs = asdict(self.cfg.calib_dataset.dataloader)
        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, **dataloader_kwargs
        )
        logger.info(f"Dataloader prepared with dataset: {type(dataset).__name__}")

    def build_pipeline(self):
        logger.info("Building execution pipeline...")

        if hasattr(self.cfg, "sparse") and self.cfg.sparse:
            ignore_layers = self.get_ignore_layers(compressor="sparse")
            self.pipeline.append(
                SparseTask(
                    model=self.model,
                    config=self.cfg.sparse,
                    ignore_layers=ignore_layers,
                    dataloader=self.dataloader,
                )
            )

        if hasattr(self.cfg, "ptq") and self.cfg.ptq:
            ignore_layers = self.get_ignore_layers(compressor="ptq")
            self.pipeline.append(
                PTQTask(
                    model=self.model,
                    config=self.cfg.ptq,
                    ignore_layers=ignore_layers,
                    dataloader=self.dataloader,
                )
            )

        if hasattr(self.cfg, "speculative") and self.cfg.speculative:
            self.pipeline.append(
                SpeculativeTask(
                    target=self.model,
                    draft=self.draft_model,
                    config=self.cfg.speculative,
                    dataloader=self.dataloader,
                )
            )

    def get_ignore_layers(self, compressor: str = "ptq"):
        default_ignore = {"lm_head", "embed_tokens", "final_layer_norm"}
        compressor_cfg = getattr(self.cfg, compressor, None)

        if compressor_cfg is None:
            user_ignore = []
        else:
            user_ignore = getattr(compressor_cfg, "ignore_layers", []) or []

        final_ignore = default_ignore.union(set(user_ignore))
        formatted_list = [f"    - {layer}" for layer in sorted(list(final_ignore))]
        layers_str = "\n".join(formatted_list)
        logger.info(f"Layers ignored during {compressor.upper()}: \n{layers_str}")
        return final_ignore

    def run(self):
        if not self.pipeline:
            logger.warning("Pipeline is empty. Nothing to run.")
            return

        for task in self.pipeline:
            task_name = type(task).__name__
            logger.info(f"======== Executing Task: {task_name} ========")
            task.execute()

        logger.success("All tasks in pipeline completed successfully.")

    def save(self):
        self.saver.save(model=self.model, pipeline=self.pipeline)
