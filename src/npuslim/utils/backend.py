import torch
import gc


class BackendHandler:
    def __init__(self):
        if hasattr(torch, "npu") and torch.npu.is_available():
            self.name, self.device, self.module = "npu", torch.device("npu"), torch.npu
        elif torch.cuda.is_available():
            self.name, self.device, self.module = (
                "cuda",
                torch.device("cuda"),
                torch.cuda,
            )
        else:
            self.name, self.device, self.module = "cpu", torch.device("cpu"), None

    def sync(self):
        if self.module:
            self.module.synchronize()

    def empty_cache(self):
        if self.module:
            self.module.empty_cache()

    def full_vacuum(self):
        if self.module:
            self.sync()
            gc.collect()
            self.empty_cache()


bh = BackendHandler()
