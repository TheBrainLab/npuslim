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
class WikiText2Dataset(BaseDataset):
    """WikiText2 dataset for calibration, matching original QuIP/GPTQ behavior."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        from datasets import load_dataset

        num_samples = self.config.num_samples
        seed = getattr(self.config, "seed", 0)
        seqlen = self.max_length

        # Load WikiText2 dataset
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

        # Tokenize all text
        text = "\n\n".join(traindata['text'])
        trainenc = self.processor(text, return_tensors='pt')

        random.seed(seed)

        for _ in range(num_samples):
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
