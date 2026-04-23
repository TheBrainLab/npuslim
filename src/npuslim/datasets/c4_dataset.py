import os
import random
from pathlib import Path

import torch

from .base_dataset import BaseDataset
from npuslim.core import DatasetRegistry


@DatasetRegistry.register("C4")
class C4Dataset(BaseDataset):
    """C4 dataset for calibration, matching original QuIP/GPTQ behavior."""

    def __init__(self, *args, seed: int = 0, **kwargs):
        self.seed = seed
        super().__init__(*args, **kwargs)
        self._load_data()

    @staticmethod
    def _cache_roots() -> list[Path]:
        roots: list[Path] = []

        hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        if hf_datasets_cache:
            roots.append(Path(hf_datasets_cache))

        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            roots.append(Path(hf_home) / "datasets")

        roots.append(Path.home() / ".cache" / "huggingface" / "datasets")

        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root not in seen:
                deduped.append(root)
                seen.add(root)
        return deduped

    def _try_remote_or_standard_cache(self):
        from datasets import load_dataset
        from loguru import logger

        load_errors: list[str] = []
        for streaming in (True, False):
            mode = "streaming" if streaming else "non-streaming"
            for config_name in ("en", None):
                display_name = config_name or "default"
                try:
                    logger.info(f"Loading C4 dataset ({mode}) with config: {display_name}")
                    load_kwargs = dict(
                        path="allenai/c4",
                        split="train",
                        streaming=streaming,
                        trust_remote_code=True,
                    )
                    if config_name is not None:
                        load_kwargs["name"] = config_name
                    ds = load_dataset(**load_kwargs)
                    if streaming:
                        return ds, load_errors
                    return iter(ds), load_errors
                except Exception as exc:
                    load_errors.append(f"{mode}/{display_name}: {exc}")
                    logger.warning(f"Failed C4 load ({mode}, config={display_name}): {exc}")
        return None, load_errors

    def _find_local_arrow_shards(self) -> list[Path]:
        candidate_dirs: list[Path] = []
        for cache_root in self._cache_roots():
            if not cache_root.exists():
                continue
            pattern = "allenai___c4/*/*/*/c4-train-*.arrow"
            for shard in cache_root.glob(pattern):
                candidate_dirs.append(shard.parent)

        if not candidate_dirs:
            return []

        newest_dir = max(candidate_dirs, key=lambda p: p.stat().st_mtime)
        return sorted(newest_dir.glob("c4-train-*.arrow"))

    def _try_local_arrow_cache(self):
        from datasets import Dataset, concatenate_datasets
        from loguru import logger

        shards = self._find_local_arrow_shards()
        if not shards:
            return None

        logger.info(f"Falling back to local C4 arrow cache with {len(shards)} shard(s)")
        parts = [Dataset.from_file(str(shard)) for shard in shards]
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        return iter(ds)

    def _load_data(self):
        from loguru import logger

        traindata, errors = self._try_remote_or_standard_cache()
        if traindata is None:
            traindata = self._try_local_arrow_cache()

        if traindata is None:
            cache_hint = ", ".join(str(p) for p in self._cache_roots())
            details = "\n".join(errors[-4:])
            raise RuntimeError(
                "Failed to load C4 dataset from Hub/cache and local Arrow cache.\n"
                f"Checked cache roots: {cache_hint}\n"
                f"Recent load errors:\n{details}"
            )

        random.seed(self.seed)

        count = 0
        for sample in traindata:
            if count >= self.num_samples:
                break

            text = sample["text"]
            trainenc = self.processor(text, return_tensors="pt")

            if trainenc.input_ids.shape[1] <= self.max_length:
                continue

            i = random.randint(0, trainenc.input_ids.shape[1] - self.max_length - 1)
            j = i + self.max_length
            inp = trainenc.input_ids[:, i:j]

            labels = inp.clone()
            labels[:, :-1] = -100

            self.data.append(
                {
                    "input_ids": inp.to(self.device),
                    "attention_mask": torch.ones_like(inp).to(self.device),
                    "labels": labels.to(self.device),
                }
            )
            count += 1

        if not self.data:
            logger.warning("C4 loaded successfully but no sample met max_seq_length requirement")
