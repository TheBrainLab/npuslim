import torch.nn as nn

class PTQObserver(nn.Module):
    def __init__(
        self,
        weight_observer=None,
        act_observer=None,
        kv_cache_observer=None,
        smooth_act_observer=None,
    ):
        super().__init__()
        self.weight_observer = weight_observer
        self.act_observer = act_observer
        self.kv_cache_observer = kv_cache_observer
        self.smooth_act_observer = smooth_act_observer

    def forward(self, input, output):
        if self.act_observer is not None:
            self.act_observer(input)
        
        if self.kv_cache_observer is not None:
            self.kv_cache_observer(output)
            
        if self.smooth_act_observer is not None:
            self.smooth_act_observer(output)