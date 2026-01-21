from dataclasses import asdict
from typing import List, Optional, TYPE_CHECKING
from loguru import logger
import fnmatch
import re
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

    @staticmethod
    def expand_patterns_to_names(all_module_names: list, patterns: list) -> list:
        """
        将通配符模式或正则表达式列表转换为模型中实际存在的层名列表。
        支持：
        1. 精确匹配: 'lm_head'
        2. 通配符匹配: '*.mlp.gate'
        3. 正则表达式: 're:.*\.mlp\.gate'
        """
        expanded_set = set()
        for pattern in patterns:
            matched = []
        
            if pattern.startswith("re:"):
                regex_str = pattern[3:]
                try:
                    reg = re.compile(regex_str)
                    matched = [name for name in all_module_names if reg.fullmatch(name)]
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{regex_str}': {e}")
                    continue

            else:
                if pattern in all_module_names:
                    matched = [pattern]
                else:
                    matched = fnmatch.filter(all_module_names, pattern)

            if matched:
                expanded_set.update(matched)
            else:
                logger.warning(
                    f"Layer pattern '{pattern}' did not match any layers in the model."
                )

        return sorted(list(expanded_set))

    def get_ignore_layers(self, compressor: str = "ptq") -> List[str]:
        all_patterns = set(self.model.skip_layer_names)
        compressor_cfg = getattr(self.cfg, compressor, None)
        if compressor_cfg is not None:
            user_ignore = getattr(compressor_cfg, "ignore_layers", []) or []
            all_patterns.update(user_ignore)

        all_leaf_names = [
            name
            for name, m in self.model.model.named_modules()
            if len(list(m.children())) == 0 and name
        ]
        final_ignore_list = self.expand_patterns_to_names(
            all_leaf_names, list(all_patterns)
        )

        if final_ignore_list:
            formatted_list = [f"    - {layer}" for layer in final_ignore_list]
            layers_str = "\n".join(formatted_list)
            logger.info(
                f"Layers ignored during {compressor.upper()} (Expanded names): \n{layers_str}"
            )
        else:
            logger.info(f"No layers ignored during {compressor.upper()}.")

        return final_ignore_list

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
