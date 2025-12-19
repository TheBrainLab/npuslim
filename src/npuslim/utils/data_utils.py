import torch
from typing import Union, Dict, List, Tuple, Any
TensorContainer = Union[Dict[str, Any], List[Any], Tuple[Any, ...]]

def batch_to_device(data: "TensorContainer", device: torch.device) -> "TensorContainer":
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: batch_to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [batch_to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(batch_to_device(v, device) for v in data)
    return data