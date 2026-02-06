from typing import Dict, List, Optional, Any, Callable
import torch.nn as nn
from ..observers import ParentObserver, PTQObserver  # 导入精简版容器


class PTQObserverHook:
    def __init__(
        self,
        model,
        observer_layers: Dict[str, nn.Module],
        act_observer: Optional[Callable] = None,
        weight_observer: Optional[Callable] = None,
        kv_cache_observer: Optional[Callable] = None,
        kv_names: Optional[List[str]] = None,
    ):
        self.quant_model = model
        self.observer_layers = observer_layers
        self.kv_names = set(kv_names) if kv_names else set()
        self._forward_hook_list = []

        # 这里的参数是工厂函数 (partial)
        self.factories = {
            "act": act_observer,
            "weight": weight_observer,
            "kv": kv_cache_observer,
        }

        # 存储结构: {layer_module: PTQObserver_instance}
        self.observer_dict: Dict[nn.Module, PTQObserver] = {}

    def apply_hook(self):
        quant_parent_dict = self.quant_model.get_parent_dict(self.observer_layers)
        parent_observers = {
            v: ParentObserver() for v in set(quant_parent_dict.values())
        }

        for name, sub_layer in self.observer_layers.items():
            # 1. 为当前层生产具体的 Observer 实例
            extra_kwargs = (
                {"parent_observer": parent_observers[quant_parent_dict[name]]}
                if name in quant_parent_dict
                else {}
            )

            # 实例化
            obs_instances = {}
            for obs_type, factory in self.factories.items():
                if factory is None:
                    continue
                if obs_type == "kv" and name not in self.kv_names:
                    continue
                # 调用 partial 工厂
                obs_instances[obs_type] = factory(sub_layer, **extra_kwargs)

            # 2. 组装进容器 (PTQObserver)
            # 这样就把字典结构扁平化为一个 Module
            container = PTQObserver(
                weight_observer=obs_instances.get("weight"),
                act_observer=obs_instances.get("act"),
                kv_cache_observer=obs_instances.get("kv"),
            )

            self.observer_dict[sub_layer] = container

            # 3. 注册 Hook
            handle = sub_layer.register_forward_hook(self._forward_hook)
            self._forward_hook_list.append(handle)

    def _forward_hook(self, layer, input, output):
        container = self.observer_dict.get(layer)
        if not container:
            return output

        # 数据预处理
        x = (
            input[0].detach().clone()
            if isinstance(input, tuple)
            else input.detach().clone()
        )
        y = (
            output[0].detach().clone()
            if isinstance(output, tuple)
            else output.detach().clone()
        )

        if hasattr(self.quant_model, "apply_layer_norm_list"):
            if layer in self.quant_model.apply_layer_norm_list:
                x = self.quant_model.apply_layer_norm(layer, x)

        # 直接调用容器的 forward，内部自动分发给各个独立的 Observer
        container(x, y)

        return output

    def remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []
    
    def post_process(self): ...
