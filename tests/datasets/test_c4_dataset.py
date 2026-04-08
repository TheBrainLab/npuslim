import sys
import types
from pathlib import Path

import torch

from npuslim.datasets.c4_dataset import C4Dataset


class DummyProcessor:
    def __call__(self, text, return_tensors):
        _ = text, return_tensors
        return types.SimpleNamespace(input_ids=torch.arange(64).unsqueeze(0))


class FakeArrowDataset:
    def __init__(self, rows):
        self._rows = rows

    def shuffle(self, seed):
        _ = seed
        return self

    def __iter__(self):
        return iter(self._rows)


def _mock_datasets_module(monkeypatch, error_message, loaded_paths):
    def fake_load_dataset(**kwargs):
        _ = kwargs
        raise ValueError(error_message)

    class FakeDatasetClass:
        @staticmethod
        def from_file(path):
            loaded_paths.append(path)
            return FakeArrowDataset([{"text": "hello " * 64}])

    def fake_concatenate_datasets(parts):
        rows = []
        for part in parts:
            rows.extend(list(part))
        return FakeArrowDataset(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            load_dataset=fake_load_dataset,
            Dataset=FakeDatasetClass,
            concatenate_datasets=fake_concatenate_datasets,
        ),
    )


def _prepare_cache_tree(cache_root: Path):
    shard = (
        cache_root
        / "allenai___c4"
        / "en-a3e66ef7800043cd"
        / "0.0.0"
        / "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
        / "c4-train-00000-of-00001.arrow"
    )
    shard.parent.mkdir(parents=True)
    shard.touch()
    return shard


def test_c4_falls_back_to_local_arrow_cache_via_hf_datasets_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "datasets_cache"
    expected_shard = _prepare_cache_tree(cache_root)
    loaded_paths = []

    _mock_datasets_module(
        monkeypatch,
        "Couldn't find cache for allenai/c4 for config 'en'\n"
        "Available configs in the cache: ['en-a3e66ef7800043cd']",
        loaded_paths,
    )
    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_root))
    monkeypatch.delenv("HF_HOME", raising=False)

    ds = C4Dataset(processor=DummyProcessor(), num_samples=1, max_seq_length=16, device="cpu")

    assert len(ds) == 1
    assert loaded_paths == [str(expected_shard)]


def test_c4_falls_back_to_local_arrow_cache_via_hf_home(monkeypatch, tmp_path):
    hf_home = tmp_path / "hf_home"
    expected_shard = _prepare_cache_tree(hf_home / "datasets")
    loaded_paths = []

    _mock_datasets_module(
        monkeypatch,
        "Couldn't find cache for allenai/c4\n"
        "Available configs in the cache: ['en-a3e66ef7800043cd']",
        loaded_paths,
    )
    monkeypatch.delenv("HF_DATASETS_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(hf_home))

    ds = C4Dataset(processor=DummyProcessor(), num_samples=1, max_seq_length=16, device="cpu")

    assert len(ds) == 1
    assert loaded_paths == [str(expected_shard)]
