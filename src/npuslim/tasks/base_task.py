from abc import ABC, abstractmethod
from pathlib import Path

class BaseTask(ABC):
    @abstractmethod
    def execute(self):
        pass

    def save(self, save_path: str | Path):
        pass