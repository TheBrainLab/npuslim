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

from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import ProcessorMixin
# from ..utils.config_parser import DatasetConfig
from npuslim.utils.config_parser import DatasetConfig


class BaseDataset(Dataset):
    """Base class for all datasets"""

    def __init__(
        self,
        *args,
        processor: "ProcessorMixin",
        config: "DatasetConfig",
        **kwargs
    ):
        self.processor = processor
        self.config = config
        self.device = config.device
        self.max_length = config.max_seq_length
        self.data = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for batching"""
        collated = {}

        # Get all keys from the first sample
        for key in batch[0].keys():
            # Collect the values for this key from all samples
            values = [item[key] for item in batch]

            # If the value is numeric, convert directly to tensor
            if isinstance(values[0], torch.Tensor):
                if all(t.dim() == 0 for t in values):  # Handle scalar tensors
                    collated[key] = torch.stack(values)
                else:
                    collated[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], (int, float)):
                collated[key] = torch.tensor(values)
            else:
                # For text and other non-numeric types, keep as a list
                collated[key] = values

        return collated