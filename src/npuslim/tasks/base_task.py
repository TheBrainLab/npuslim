from abc import ABC, abstractmethod
from pathlib import Path
    

class BaseTask(ABC):
    @abstractmethod
    def execute(self): ...
    
    @abstractmethod
    def save_model(self, save_path: Path | str): ...
        
    @abstractmethod
    def save_meta(self, save_path: Path | str): ...
