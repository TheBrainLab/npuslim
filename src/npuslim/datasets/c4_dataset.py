import random

import torch

from .base_dataset import BaseDataset
from npuslim.registry import DatasetRegistry


@DatasetRegistry.register("C4")
class C4Dataset(BaseDataset):
    """C4 dataset for calibration, matching original QuIP/GPTQ behavior."""

    def __init__(self, *args, num_samples: int = 256, seed: int = 0, **kwargs):
        self.num_samples = num_samples
        self.seed = seed
        super().__init__(*args, **kwargs)
        self._load_data()

    def _load_data(self):
        from datasets import load_dataset
        from loguru import logger

        num_samples = self.num_samples
        seed = self.seed
        seqlen = self.max_length

        # Try loading C4 dataset with 'en' config (most common)
        # If cache mismatch error occurs, auto-detect available configs from error
        traindata = None
        configs_to_try = ['en', None]

        for config_name in configs_to_try:
            try:
                load_kwargs = {
                    'path': 'allenai/c4',
                    'split': 'train',
                    'streaming': True,
                    'trust_remote_code': True,
                }
                if config_name is not None:
                    load_kwargs['name'] = config_name

                logger.info(f"Loading C4 dataset with config: {config_name or 'default'}")
                traindata = load_dataset(**load_kwargs)
                logger.info(f"Successfully loaded C4 dataset")
                break
            except ValueError as e:
                # Check if this is a cache mismatch error with available configs listed
                error_msg = str(e)
                if "Available configs in the cache" in error_msg:
                    # Extract available configs from error message
                    import re
                    match = re.search(r"Available configs in the cache: \[(.*?)\]", error_msg)
                    if match:
                        available = [c.strip().strip("'\"") for c in match.group(1).split(",")]
                        logger.warning(f"Cache config mismatch. Available configs: {available}")
                        # Try the first available config (most likely to be correct)
                        if available:
                            logger.info(f"Retrying with cached config: {available[0]}")
                            try:
                                traindata = load_dataset(
                                    'allenai/c4',
                                    available[0],
                                    split='train',
                                    streaming=True,
                                    trust_remote_code=True
                                )
                                logger.info(f"Successfully loaded C4 with cached config")
                                break
                            except Exception as e2:
                                logger.warning(f"Failed with cached config: {e2}")
                logger.warning(f"Failed to load C4 with config '{config_name or 'default'}': {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to load C4 with config '{config_name or 'default'}': {e}")
                continue

        if traindata is None:
            raise RuntimeError(
                "Failed to load C4 dataset. Please check your network connection, "
                "or try clearing the cache with: rm -rf ~/.cache/huggingface/datasets"
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
