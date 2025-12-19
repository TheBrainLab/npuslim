from abc import ABC, abstractmethod
from dataclasses import asdict
from torch.utils.data import DataLoader

from npuslim.utils.config_parser import GlobalConfig
from npuslim.utils.factory import ModelFactory, DatasetFactory


class BaseEngine(ABC):
    def __init__(self):
        self.cfg = GlobalConfig.get_config()
        self.slim_model = None
        self.dataloader = None

        self.prepare_model()
        self.prepare_dataloader()

    def prepare_model(self):
        model_cfg = self.cfg.model
        self.slim_model = ModelFactory.create(config=model_cfg)
        self.slim_model.prepare()

    def prepare_dataloader(self):
        if self.cfg.calib_dataset is None:
            return

        # dataset_kwargs = dict(self.cfg.calib_dataset.dataset)
        processor = (
            self.slim_model.processor
            if self.cfg.meta.type in ["vlm"]
            else self.slim_model.tokenizer
        )
        dataset = DatasetFactory.create(
            processor=processor, config=self.cfg.calib_dataset.dataset
        )

        dataloader_kwargs = asdict(self.cfg.calib_dataset.dataloader)
        self.dataloader = DataLoader(
            dataset, collate_fn=dataset.collate_fn, **dataloader_kwargs
        )

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def save(self): ...
