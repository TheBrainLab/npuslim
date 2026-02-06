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
import numpy as np
import torch
from datasets import load_dataset
from typing import Dict, List

from .base_dataset import BaseDataset
from npuslim.utils.factory import DatasetFactory


@DatasetFactory.register()
class MMLUDataset(BaseDataset):
    """Dataset for MMLU reasoning-style calibration data"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 从 self.config 中读取参数，确保框架兼容性
        num_samples = getattr(self.config, "num_samples", 128)
        seed = getattr(self.config, "seed", 42)
        data_path = self.config.data_path
        self._load_mmlu_data(data_path, num_samples, seed)

    def _load_mmlu_data(self, data_path: str, num_samples: int, seed: int):
        # 1. 固定随机种子以保证实验可复现性
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"Loading MMLU validation set and sampling {num_samples} items with seed {seed}...")

        # 2. 加载 MMLU 验证集 (包含 57 个学科)
        try:
            # 优先从本地缓存或 Hub 加载
            dataset = load_dataset(data_path, "all", split="validation")
        except Exception as e:
            print(f"Error loading MMLU dataset: {e}. Please ensure 'datasets' is up to date.")
            return

        # 3. 随机采样
        total_len = len(dataset)
        sample_size = min(num_samples, total_len) if num_samples > 0 else total_len
        indices = random.sample(range(total_len), sample_size)

        choices_map = ["A", "B", "C", "D"]

        # 4. 构造 Prompt 并进行 Tokenization
        for idx in indices:
            item = dataset[idx]
            question = item['question']
            choices = item['choices']
            answer_idx = item['answer']  # 0, 1, 2, 3

            # 构造符合 RQP 方案的 Reasoning Prompt
            prompt = (
                f"The following are multiple choice questions (with answers).\n\n"
                f"Question: {question}\n"
            )
            for i, choice in enumerate(choices):
                prompt += f"{choices_map[i]}. {choice}\n"
            prompt += "Answer:"

            # 使用框架自带的 processor (tokenizer)
            model_inputs = self.processor(
                [prompt],
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )

            # 构造 Labels
            # 如果你有 labels 需求，可以存入正确答案的 Token，或者按照 TextDataset 逻辑 roll
            labels = model_inputs["input_ids"].roll(shifts=-1, dims=-1)
            labels[:, -1] = -100

            # 这里的 labels 也可以选择存储正确答案的 ID，方便做 Supervised RQP
            # target_char = choices_map[answer_idx]

            data_item = {
                "input_ids": model_inputs["input_ids"].to(self.device),
                "attention_mask": model_inputs["attention_mask"].to(self.device),
                "labels": labels.to(self.device),
            }
            self.data.append(data_item)

        print(f"Successfully loaded {len(self.data)} MMLU reasoning samples.")