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

import random

import torch

from .base_dataset import BaseDataset
from npuslim.utils.factory import DatasetFactory


@DatasetFactory.register()
class C4Dataset(BaseDataset):
    """C4 dataset for calibration, matching original QuIP/GPTQ behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        from datasets import load_dataset

        num_samples = self.config.num_samples
        seed = getattr(self.config, "seed", 0)
        seqlen = self.max_length

        # Load C4 dataset (matching original QuIP/GPTQ)
        # Use 'en' config for English subset with streaming
        traindata = load_dataset(
            'allenai/c4',
            'en',
            split='train',
            streaming=True,
            trust_remote_code=True
        )

        random.seed(seed)

        count = 0
        for sample in traindata:
            if count >= num_samples:
                break

            text = sample['text']
            trainenc = self.processor(
                text,
                return_tensors='pt'
            )

            # Skip if too short
            if trainenc.input_ids.shape[1] <= seqlen:
                continue

            # Random slice of sequence length
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]

            # Create labels (shifted, with -100 for first token)
            labels = inp.clone()
            labels[:, :-1] = -100

            self.data.append({
                "input_ids": inp.to(self.device),
                "attention_mask": torch.ones_like(inp).to(self.device),
                "labels": labels.to(self.device),
            })
            count += 1
