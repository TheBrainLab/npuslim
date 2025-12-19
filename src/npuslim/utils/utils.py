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