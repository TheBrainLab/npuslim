from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from npuslim.utils.config_parser import Configuration
from npuslim.utils.factory import ModelFactory, DatasetFactory


class BaseEngine(ABC):
    def __init__(self):
        self.cfg = Configuration.prepare()
        self.slim_type = self.cfg.metadata.type.lower()
        assert self.slim_type in [
            "llm",
            "vlm",
        ], f"Unsupported engine type: {self.cfg.metadata.type}. Must be 'llm' or 'vlm'."
        assert (
            "model" in self.cfg and self.cfg.model
        ), "Missing 'model' configuration in YAML."
        assert (
            "calib_dataset" in self.cfg and self.cfg.calib_dataset
        ), "Missing 'calib_dataset' configuration in YAML."

        self.slim_model = None
        self.dataloader = None

        self.prepare_model()
        self.prepare_dataloader()

    def prepare_model(self):
        model_cfg = dict(self.cfg.model)
        self.slim_model = ModelFactory.create(**model_cfg)
        self.slim_model.prepare()

    def prepare_dataloader(self):
        dataset_cfg = dict(self.cfg.calib_dataset.dataset)
        dataloader_cfg = dict(self.cfg.calib_dataset.dataloader)
        processor = (
            self.slim_model.processor
            if self.slim_type in ["vlm"]
            else self.slim_model.tokenizer
        )
        dataset = DatasetFactory.create(processor=processor, **dataset_cfg)
        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, **dataloader_cfg
        )

    def save(self): ...

    @abstractmethod
    def run(self): ...
