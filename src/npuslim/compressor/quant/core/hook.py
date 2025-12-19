# Copyright 2025 Tencent Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..observers import PTQObserver, ParentObserver
# from ..core.quant_algo_info import QuantConfigManager


class PTQObserverHook:
    def __init__(self, quant_model, quant_info):
        self.quant_model = quant_model
        self._forward_hook_list = []
        # {name: layer}
        self.quant_layers_dict = {}
        # {layer: observer}
        self.observer_dict = {}
        self.kv_names = []
        self.quant_algo_info = quant_info

    def apply_hook(self):
        self.quant_layers_dict = self.quant_algo_info.observer_layers_names
        self.kv_names = self.quant_algo_info.kv_names

        act_observer = self.quant_algo_info.act_observer
        weight_observer = self.quant_algo_info.weight_observer
        kv_cache_observer = self.quant_algo_info.kv_cache_observer

        quant_parent_dict = self.quant_model.get_parent_dict(self.quant_layers_dict)
        parent_observers = {
            v: ParentObserver() for v in set(quant_parent_dict.values())
        }

        # apply observers
        for name, sub_layer in self.quant_layers_dict.items():
            extra_kwargs = (
                {"parent_observer": parent_observers[quant_parent_dict[name]]}
                if name in quant_parent_dict
                else {}
            )
            observer = PTQObserver(
                sub_layer,
                act_observer,
                weight_observer,
                kv_cache_observer if name in self.kv_names else None,
                **extra_kwargs,
            )
            forward_hook_handle = sub_layer.register_forward_hook(self._forward_hook)
            self.observer_dict[sub_layer] = observer
            self._forward_hook_list.append(forward_hook_handle)

    def _forward_hook(self, layer, input, output):
        x = input[0].clone() if isinstance(input, tuple) else input.clone()
        y = output[0].clone() if isinstance(output, tuple) else output.clone()
        if hasattr(self.quant_model, "apply_layer_norm_list"):
            if layer in self.quant_model.apply_layer_norm_list:
                x = self.quant_model.apply_layer_norm(layer, x)
        self.observer_dict[layer](x, y)
        return output

    def remove_hook(self):
        for hook in self._forward_hook_list:
            hook.remove()
        self._forward_hook_list = []

    def post_process(self): ...