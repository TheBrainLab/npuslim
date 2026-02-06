from typing import Dict, Any, Type, Union, TypeVar
from dataclasses import fields, is_dataclass
import torch


def find_layers(module, layers=None, name=""):
    if not layers:
        layers = [torch.nn.Linear]
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child,
                layers=layers,
                name=name + "." + name1 if name != "" else name1,
            )
        )
    return res


def find_parent_layer_and_sub_name(model, name):
    last_idx = 0
    idx = 0
    parent_layer = model
    while idx < len(name):
        if name[idx] == ".":
            sub_name = name[last_idx:idx]
            if hasattr(parent_layer, sub_name):
                parent_layer = getattr(parent_layer, sub_name)
                last_idx = idx + 1
        idx += 1
    sub_name = name[last_idx:idx]
    return parent_layer, sub_name


T = TypeVar("T")


def create_or_update_dataclass(target: Union[Type[T], T], data: Dict[str, Any]) -> T:
    if is_dataclass(target) and isinstance(target, type):
        is_instance = False
        cls = target
    elif is_dataclass(target) and not isinstance(target, type):
        is_instance = True
        cls = type(target)
    else:
        raise ValueError(f"Target {target} must be a dataclass type or instance.")

    if not data:
        return target if is_instance else cls()

    valid_fields = {f.name for f in fields(cls)}
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}

    if is_instance:
        for k, v in filtered_data.items():
            setattr(target, k, v)
        return target
    else:
        return cls(**filtered_data)
