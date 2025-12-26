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
# Modified by weiyangdaren.

from typing import TYPE_CHECKING, Dict, Any
from ...helper.hook import HooksMixin
from ..observers import PTQObserver, ParentObserver


class PTQObserverHook(HooksMixin):

    quant_model: Any
    quant_algo_info: Any
    observer_dict: Dict[Any, Any] = {}

    def apply_hook(self):
        observer_layers_names = self.quant_algo_info.observer_layers_names
        kv_names = self.quant_algo_info.kv_names

        act_observer = self.quant_algo_info.act_observer
        weight_observer = self.quant_algo_info.weight_observer
        kv_cache_observer = self.quant_algo_info.kv_cache_observer

        quant_parent_dict = self.quant_model.get_parent_dict(observer_layers_names)
        parent_observers = {
            v: ParentObserver() for v in set(quant_parent_dict.values())
        }

        for name, sub_layer in observer_layers_names.items():
            extra_kwargs = (
                {"parent_observer": parent_observers[quant_parent_dict[name]]}
                if name in quant_parent_dict
                else {}
            )
            observer = PTQObserver(
                sub_layer,
                act_observer,
                weight_observer,
                kv_cache_observer if name in kv_names else None,
                **extra_kwargs,
            )
            self.observer_dict[sub_layer] = observer
            self.register_hook(sub_layer, self._forward_hook, hook_type="forward")

    def _forward_hook(self, layer, input, output):
        x = input[0].clone() if isinstance(input, tuple) else input.clone()
        y = output[0].clone() if isinstance(output, tuple) else output.clone()

        if hasattr(self.quant_model, "apply_layer_norm_list"):
            if layer in self.quant_model.apply_layer_norm_list:
                x = self.quant_model.apply_layer_norm(layer, x)

        self.observer_dict[layer](x, y)
        return output

    def post_process(self): ...
