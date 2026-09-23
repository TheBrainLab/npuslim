# tests/tasks/compressor/test_resume.py
"""Unit tests for resume (automatic checkpoint) support.

Covers:
- StreamingHuggingFaceSaver.recover_from_disk / resume_manifest / shard matcher
- CompressorTask config fingerprint (stability + sensitivity)
- BaseHessianAlgorithm.save_resume_state / load_resume_state roundtrip
- CompressorTask checkpoint commit / restore validation (_try_resume)
- End-to-end interrupted run -> resume -> identical output (fake checkpoint)
- End-to-end config-drift rejection (fingerprint mismatch)
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from npuslim.algorithms.quantization.hessian.base_hessian_algo import (
    BaseHessianAlgorithm,
)
from npuslim.core.backend import bh
from npuslim.savers.hf_saver import StreamingHuggingFaceSaver
from npuslim.tasks.compressor.task import CompressorTask


# =============================================================================
# Fixtures / helpers
# =============================================================================

def _make_fake_checkpoint(model_dir: Path, num_layers: int = 4) -> set:
    """Create a minimal sharded safetensors checkpoint (2 tensors per layer)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.float32),
        "lm_head.weight": torch.randn(32, 16, dtype=torch.float32),
    }
    for i in range(num_layers):
        tensors[f"model.layers.{i}.fc1.weight"] = torch.randn(8, 16, dtype=torch.float32)
        tensors[f"model.layers.{i}.fc2.weight"] = torch.randn(16, 8, dtype=torch.float32)

    shard_name = "model-00000.safetensors"
    save_file(tensors, str(model_dir / shard_name))
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {name: shard_name for name in tensors},
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    return set(tensors.keys())


class FakeModelObj:
    """Minimal model wrapper satisfying CompressorTask._create_loader / run."""

    def __init__(self, path_str: str, num_layers: int):
        self.path_str = path_str
        self.path = path_str
        self.model_hub = "hf"
        self.model_kwargs = {}
        self.block_name = "model.layers"
        self.pre_transformer_module_names = ["model.embed_tokens"]
        self.post_transformer_module_names = ["lm_head"]
        self.num_transformer_layers = num_layers
        self.skip_layer_names = []
        self.mtp_layer_names = []
        self.config = None
        self.tokenizer = None
        self.processor = None


class FakeAlgo:
    """Stub algorithm with pluggable interruption and real state files."""

    _quantize_mtp = False
    _save_mtp_debug = False
    max_calib_samples = 8

    def __init__(self, fail_at=None):
        self.fail_at = fail_at
        self.processed_chunks = []
        self.saved_states = []
        self.loaded_states = []
        self.next_layer = 0

    def on_start(self):
        pass

    def on_finish(self):
        pass

    def process_chunk(self, chunk):
        self.processed_chunks.append(chunk.chunk_index)
        if self.fail_at is not None and chunk.chunk_index >= self.fail_at:
            raise RuntimeError(f"simulated interruption at chunk {chunk.chunk_index}")
        self.next_layer = chunk.layers[-1].index + 1
        return chunk

    def save_resume_state(self, path):
        torch.save({"version": 1, "next_layer": self.next_layer}, path)
        self.saved_states.append(Path(path).name)

    def load_resume_state(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.loaded_states.append(Path(path).name)
        return payload["next_layer"]


def _make_task(
    model_dir: Path,
    output_dir: Path,
    *,
    num_layers: int = 4,
    chunk_size: int = 1,
    resume: bool = True,
    algo=None,
):
    task = CompressorTask(
        name="resume_test",
        model="@fake",
        algorithm={"type": "Fake", "wbits": 4},
        execution={"mode": "streaming", "chunk_size": chunk_size, "resume": resume},
        resource_manager=MagicMock(),
    )
    task._model_obj = FakeModelObj(str(model_dir), num_layers)
    task._algorithm = algo if algo is not None else FakeAlgo()
    # Large threshold so one chunk == one shard (mirrors real runs with 4GiB).
    task._saver = StreamingHuggingFaceSaver(
        output_dir=output_dir, size_threshold=1024**3
    )
    return task


# =============================================================================
# A. Saver: shard matcher / manifest / recover_from_disk
# =============================================================================

def test_shard_matcher_extracts_numeric_index():
    matcher = StreamingHuggingFaceSaver._make_shard_matcher("model-{:05d}.safetensors")
    assert matcher.match("model-00001.safetensors").group(1) == "00001"
    # Matching is intentionally lenient about digit width; unrelated names are rejected.
    assert matcher.match("model-1.safetensors").group(1) == "1"
    assert matcher.match("other-00001.safetensors") is None

    custom = StreamingHuggingFaceSaver._make_shard_matcher("shard_{:03d}.st")
    assert custom.match("shard_042.st").group(1) == "042"

    with pytest.raises(ValueError, match="numeric format field"):
        StreamingHuggingFaceSaver._make_shard_matcher("no-field.safetensors")


def _saver_with_two_shards(output_dir: Path):
    saver = StreamingHuggingFaceSaver(output_dir=output_dir, size_threshold=1024)
    saver.add_tensor("a.weight", torch.ones(4, dtype=torch.float32))
    saver.add_tensor("b.weight", torch.ones(4, dtype=torch.float32))
    saver.flush()
    saver.add_tensor("c.weight", torch.ones(4, dtype=torch.float32))
    saver.flush()
    return saver


def test_recover_from_disk_restores_state_and_removes_orphans(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    manifest = saver.resume_manifest()
    assert manifest["shard_counter"] == 2
    assert set(manifest["written_shards"]) == {
        "model-00000.safetensors",
        "model-00001.safetensors",
    }

    # Simulate crash after flush but before manifest commit:
    save_file({"orphan.weight": torch.ones(2)}, str(out / "model-00002.safetensors"))
    (out / "model-00003.safetensors.tmp").write_bytes(b"interrupted")

    recovered = StreamingHuggingFaceSaver(output_dir=out, size_threshold=1024)
    summary = recovered.recover_from_disk(manifest)

    assert summary["shards"] == 2
    assert summary["tensors"] == 3
    assert summary["orphan_shards_removed"] == 1
    assert recovered.shard_counter == 2
    assert recovered.weight_map == {
        "a.weight": "model-00000.safetensors",
        "b.weight": "model-00000.safetensors",
        "c.weight": "model-00001.safetensors",
    }
    assert recovered.tensor_type_map["a.weight"] == "FLOAT"
    assert not (out / "model-00002.safetensors").exists()
    assert not (out / "model-00003.safetensors.tmp").exists()

    # New flush continues numbering after the deleted orphan.
    recovered.add_tensor("d.weight", torch.ones(4, dtype=torch.float32))
    recovered.flush()
    assert (out / "model-00002.safetensors").exists()


def test_recover_from_disk_rejects_missing_referenced_shard(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    manifest = saver.resume_manifest()
    manifest["weight_map"]["ghost.weight"] = "model-00009.safetensors"

    recovered = StreamingHuggingFaceSaver(output_dir=out, size_threshold=1024)
    with pytest.raises(IOError, match="missing shard"):
        recovered.recover_from_disk(manifest)


def test_recover_from_disk_rejects_untyped_tensors_on_npu(tmp_path, monkeypatch):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    manifest = saver.resume_manifest()
    manifest.pop("tensor_type_map")  # types cannot be rebuilt from disk on NPU

    monkeypatch.setattr(bh, "_has_npu", True)  # has_npu is a read-only property
    recovered = StreamingHuggingFaceSaver(output_dir=out, size_threshold=1024)
    with pytest.raises(ValueError, match="tensor_type"):
        recovered.recover_from_disk(manifest)


def test_manifest_and_recover_require_clean_saver_state(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    manifest = saver.resume_manifest()

    saver.add_tensor("pending.weight", torch.ones(4, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="empty buffer"):
        saver.resume_manifest()
    with pytest.raises(RuntimeError, match="before adding tensors"):
        saver.recover_from_disk(manifest)


# =============================================================================
# B. Config fingerprint
# =============================================================================

_FP_LOADER = SimpleNamespace(
    resolve_model_source=lambda: "/models/qwen",
    model_hub="hf",
    model_kwargs={},
    num_layers=4,
)
_FP_SAVER = SimpleNamespace(shard_name_pattern="model-{:05d}.safetensors")
_FP_ALGO = SimpleNamespace(max_calib_samples=8)


def _fingerprint_task(**execution_overrides):
    execution = {"mode": "streaming", "chunk_size": 1}
    execution.update(execution_overrides)
    return CompressorTask(
        name="fp",
        model="@m",
        algorithm={"type": "Fake", "wbits": 4},
        execution=execution,
        resource_manager=MagicMock(),
    )


def test_config_fingerprint_stable_and_sensitive():
    fp = lambda t, skips=(): t._config_fingerprint(_FP_LOADER, _FP_SAVER, _FP_ALGO, list(skips))  # noqa: E731

    # Deterministic for identical inputs, across task instances.
    assert fp(_fingerprint_task()) == fp(_fingerprint_task())

    # Sensitive: chunk_size, algorithm config, skip list.
    assert fp(_fingerprint_task(chunk_size=2)) != fp(_fingerprint_task())
    changed_algo = _fingerprint_task()
    changed_algo.algorithm_config["wbits"] = 8
    assert fp(changed_algo) != fp(_fingerprint_task())
    assert fp(_fingerprint_task(), skips=["model.layers.0.fc1"]) != fp(_fingerprint_task())


# =============================================================================
# C. BaseHessianAlgorithm resume state roundtrip
# =============================================================================

def test_hessian_resume_state_roundtrip(tmp_path):
    algo = BaseHessianAlgorithm()
    algo._inps = torch.randn(4, 3, 8)
    algo._prev_topk_indices = torch.zeros(4, 5, dtype=torch.int64)
    algo._layer_kwargs = {
        "attention_mask": torch.ones(4, 3),
        "position_embeddings": (torch.ones(1, 8), torch.ones(1, 8)),
        "use_cache": False,
    }
    algo._calib_batch_size = 2
    algo._next_expected_layer_index = 7
    algo._total_layers = 12

    state_path = tmp_path / "algo_state_7.pt"
    algo.save_resume_state(state_path)

    fresh = BaseHessianAlgorithm()
    assert fresh.load_resume_state(state_path) == 7
    assert torch.equal(fresh._inps, algo._inps)
    assert torch.equal(fresh._prev_topk_indices, algo._prev_topk_indices)
    assert fresh._outs.shape == algo._inps.shape
    assert float(fresh._outs.abs().sum()) == 0.0
    assert isinstance(fresh._layer_kwargs["position_embeddings"], tuple)
    assert fresh._layer_kwargs["use_cache"] is False
    assert fresh._calib_batch_size == 2
    assert fresh._total_layers == 12
    assert fresh._layer_kwargs_need_move is True


def test_hessian_load_resume_state_rejects_inconsistent_topk(tmp_path):
    algo = BaseHessianAlgorithm()
    algo._inps = torch.randn(4, 3, 8)
    algo._prev_topk_indices = torch.zeros(4, 5, dtype=torch.int64)
    algo._layer_kwargs = {}
    state_path = tmp_path / "state.pt"
    algo.save_resume_state(state_path)

    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    payload["prev_topk_indices"] = torch.zeros(9, 5, dtype=torch.int64)
    torch.save(payload, state_path)

    with pytest.raises(ValueError, match="inconsistent"):
        BaseHessianAlgorithm().load_resume_state(state_path)


# =============================================================================
# D. Task checkpoint commit / _try_resume validation
# =============================================================================

def _fake_loader_for_resume(chunk_size=1):
    return SimpleNamespace(
        get_chunk_layer_indices=lambda idx: [idx * chunk_size],
    )


def test_commit_checkpoint_writes_progress_and_prunes_state_files(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    task = _make_task(tmp_path / "model", out)

    fingerprint = "fp-abc"
    algo = FakeAlgo()
    algo.next_layer = 1
    task._commit_checkpoint(
        algo=algo, saver=saver, stage="chunks", next_chunk_idx=1,
        touched_original_keys={"a.weight"}, fingerprint=fingerprint, chunk_count=4,
    )
    algo.next_layer = 2
    task._commit_checkpoint(
        algo=algo, saver=saver, stage="chunks", next_chunk_idx=2,
        touched_original_keys={"a.weight", "b.weight"}, fingerprint=fingerprint, chunk_count=4,
    )

    resume_dir = out / ".npuslim_resume"
    state_files = sorted(p.name for p in resume_dir.glob("algo_state_*.pt"))
    assert state_files == ["algo_state_2.pt"]  # only newest kept
    assert algo.saved_states == ["algo_state_1.pt", "algo_state_2.pt"]

    progress = json.loads((resume_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "chunks"
    assert progress["next_chunk_idx"] == 2
    assert progress["chunk_count"] == 4
    assert progress["fingerprint"] == fingerprint
    assert progress["algo_state_file"] == "algo_state_2.pt"
    assert set(progress["touched_original_keys"]) == {"a.weight", "b.weight"}
    assert progress["saver_manifest"]["shard_counter"] == 2
    assert set(progress["saver_manifest"]["weight_map"]) == {"a.weight", "b.weight", "c.weight"}


def test_try_resume_restores_state_and_returns_context(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    task = _make_task(tmp_path / "model", out)

    algo = FakeAlgo()
    algo.next_layer = 1
    task._commit_checkpoint(
        algo=algo, saver=saver, stage="chunks", next_chunk_idx=1,
        touched_original_keys={"a.weight", "b.weight"}, fingerprint="fp", chunk_count=4,
    )

    fresh_task = _make_task(tmp_path / "model", out)
    fresh_algo = FakeAlgo()
    ctx = fresh_task._try_resume(
        fresh_algo, fresh_task._saver, _fake_loader_for_resume(), "fp", 4,
        {"a.weight", "b.weight", "c.weight"},
    )

    assert ctx["stage"] == "chunks"
    assert ctx["next_chunk_idx"] == 1
    assert ctx["touched"] == {"a.weight", "b.weight"}
    assert fresh_algo.loaded_states == ["algo_state_1.pt"]
    # Saver recovered into the fresh task's instance.
    assert fresh_task._saver.shard_counter == 2
    assert set(fresh_task._saver.weight_map) == {"a.weight", "b.weight", "c.weight"}


def _write_progress(resume_dir: Path, **overrides):
    progress = {
        "version": 1,
        "stage": "chunks",
        "next_chunk_idx": 1,
        "chunk_count": 4,
        "fingerprint": "fp",
        "touched_original_keys": ["a.weight"],
        "saver_manifest": {},
        "algo_state_file": "algo_state_1.pt",
    }
    progress.update(overrides)
    resume_dir.mkdir(parents=True, exist_ok=True)
    (resume_dir / "progress.json").write_text(
        json.dumps(progress), encoding="utf-8"
    )
    return progress


@pytest.mark.parametrize(
    "overrides, match",
    [
        # fingerprint drift
        ({"fingerprint": "different"}, "fingerprint mismatch"),
        # unsupported version
        ({"version": 99}, "version"),
        # manifest missing
        ({"saver_manifest": {}}, "no saver manifest"),
        # next_chunk_idx beyond chunk_count
        ({"next_chunk_idx": 9}, "out of range"),
        # touched keys unknown to the current model
        ({"touched_original_keys": ["bogus.weight"]}, "unknown to current model"),
    ],
)
def test_try_resume_rejects_invalid_progress(tmp_path, overrides, match):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    manifest = saver.resume_manifest()
    overrides = dict(overrides)
    overrides.setdefault("saver_manifest", manifest)
    _write_progress(out / ".npuslim_resume", **overrides)

    task = _make_task(tmp_path / "model", out)
    with pytest.raises(ValueError, match=match):
        task._try_resume(
            FakeAlgo(), task._saver, _fake_loader_for_resume(), "fp", 4,
            {"a.weight"},
        )


def test_try_resume_rejects_missing_state_file(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    _write_progress(
        out / ".npuslim_resume",
        saver_manifest=saver.resume_manifest(),
        algo_state_file="algo_state_99.pt",  # never written
    )

    task = _make_task(tmp_path / "model", out)
    with pytest.raises(ValueError, match="Missing algo state file"):
        task._try_resume(
            FakeAlgo(), task._saver, _fake_loader_for_resume(), "fp", 4,
            {"a.weight"},
        )


def test_try_resume_rejects_layer_boundary_mismatch(tmp_path):
    out = tmp_path / "out"
    saver = _saver_with_two_shards(out)
    state_path = out / ".npuslim_resume" / "algo_state_1.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": 1, "next_layer": 5}, state_path)  # expects layer 5
    _write_progress(out / ".npuslim_resume", saver_manifest=saver.resume_manifest())

    task = _make_task(tmp_path / "model", out)
    with pytest.raises(ValueError, match="expects next layer 5"):
        task._try_resume(
            FakeAlgo(), task._saver, _fake_loader_for_resume(), "fp", 4,
            {"a.weight"},
        )


# =============================================================================
# E. End-to-end: interrupted run -> resume -> identical output
# =============================================================================

def test_run_interrupted_then_resumes_and_finalizes(tmp_path):
    model_dir = tmp_path / "model"
    all_keys = _make_fake_checkpoint(model_dir, num_layers=4)
    out_dir = tmp_path / "out"

    # --- First run: crash while processing chunk 2 (chunks 0/1 committed). ---
    algo1 = FakeAlgo(fail_at=2)
    task1 = _make_task(model_dir, out_dir, algo=algo1)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        task1.run()

    resume_dir = out_dir / ".npuslim_resume"
    progress = json.loads((resume_dir / "progress.json").read_text(encoding="utf-8"))
    assert progress["stage"] == "chunks"
    assert progress["next_chunk_idx"] == 2
    assert progress["chunk_count"] == 4
    assert set(progress["touched_original_keys"]) == {
        "model.embed_tokens.weight",
        "model.layers.0.fc1.weight",
        "model.layers.0.fc2.weight",
        "model.layers.1.fc1.weight",
        "model.layers.1.fc2.weight",
    }
    assert sorted(p.name for p in resume_dir.glob("algo_state_*.pt")) == ["algo_state_2.pt"]

    # --- Second run: resumes from chunk 2 and completes. ---
    algo2 = FakeAlgo()
    task2 = _make_task(model_dir, out_dir, algo=algo2)
    result = task2.run()

    assert algo2.processed_chunks == [2, 3]  # skipped chunks 0/1
    assert algo2.loaded_states == ["algo_state_2.pt"]
    assert result["resumed"] is True
    assert result["resumed_from_chunk"] == 2
    assert not resume_dir.exists()  # cleaned after successful finalize

    index = json.loads((out_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    assert set(index["weight_map"]) == all_keys
    assert len(set(index["weight_map"].values())) == 4  # chunk boundary == shard boundary


def test_run_resume_rejects_config_drift_and_keeps_checkpoint(tmp_path):
    model_dir = tmp_path / "model"
    _make_fake_checkpoint(model_dir, num_layers=4)
    out_dir = tmp_path / "out"

    task1 = _make_task(model_dir, out_dir, algo=FakeAlgo(fail_at=2))
    with pytest.raises(RuntimeError, match="simulated interruption"):
        task1.run()

    progress_path = out_dir / ".npuslim_resume" / "progress.json"
    assert progress_path.exists()

    # Same output dir but chunk_size changed -> fingerprint mismatch.
    task2 = _make_task(model_dir, out_dir, chunk_size=2)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        task2.run()

    # Checkpoint untouched by the rejected run.
    assert progress_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["next_chunk_idx"] == 2


def test_run_resume_rejects_mtp_options(tmp_path):
    model_dir = tmp_path / "model"
    _make_fake_checkpoint(model_dir, num_layers=4)
    out_dir = tmp_path / "out"

    task = _make_task(model_dir, out_dir)
    task._algorithm._save_mtp_debug = True
    with pytest.raises(ValueError, match="quantize_mtp/save_mtp_debug"):
        task.run()
