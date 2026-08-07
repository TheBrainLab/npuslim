#!/usr/bin/env python
"""
Mock test: verify that layer 78 (MTP) is correctly saved via backfill
when quantize_mtp=False, save_mtp_debug=False.

This test simulates the CompressorTask MTP/backfill logic without requiring
a real GLM-5 model or GPU. It creates a mock ChunkLoader and saver, then
verifies that layer 78 tensors end up in the output as FLOAT.

Usage:
    conda run -n npuslim python tools/test_mtp_backfill.py
"""
import json
import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock, patch
from typing import Dict, List, Set

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class MockChunkLoader:
    """Mock loader that simulates GLM-5 checkpoint with 79 regular layers + 1 MTP layer."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        # Simulate layer 78 (MTP) tensors: a few small tensors
        self.layer78_tensors = {
            "model.layers.78.eh_proj.weight": torch.randn(6144, 6144, dtype=torch.bfloat16),
            "model.layers.78.eh_proj.bias": torch.randn(6144, dtype=torch.bfloat16),
            "model.layers.78.experts.0.gate_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.experts.0.up_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.experts.0.down_proj.weight": torch.randn(6144, 7168, dtype=torch.bfloat16),
            "model.layers.78.experts.1.gate_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.experts.1.up_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.experts.1.down_proj.weight": torch.randn(6144, 7168, dtype=torch.bfloat16),
            "model.layers.78.moe_shared_experts.gate_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.moe_shared_experts.up_proj.weight": torch.randn(7168, 6144, dtype=torch.bfloat16),
            "model.layers.78.moe_shared_experts.down_proj.weight": torch.randn(6144, 7168, dtype=torch.bfloat16),
            "model.layers.78.self_attention.q_a_proj.weight": torch.randn(960, 6144, dtype=torch.bfloat16),
            "model.layers.78.self_attention.q_b_proj.weight": torch.randn(6144, 960, dtype=torch.bfloat16),
            "model.layers.78.self_attention.kv_a_proj_with_mqa.weight": torch.randn(576, 6144, dtype=torch.bfloat16),
            "model.layers.78.self_attention.kv_b_proj.weight": torch.randn(6144, 576, dtype=torch.bfloat16),
            "model.layers.78.self_attention.o_proj.weight": torch.randn(6144, 6144, dtype=torch.bfloat16),
            "model.layers.78.post_attention_layernorm.weight": torch.randn(6144, dtype=torch.bfloat16),
            "model.layers.78.post_attention_layernorm.bias": torch.randn(6144, dtype=torch.bfloat16),
            "model.layers.78.input_layernorm.weight": torch.randn(6144, dtype=torch.bfloat16),
            "model.layers.78.input_layernorm.bias": torch.randn(6144, dtype=torch.bfloat16),
        }
        # Simulate regular layer tensors (layer 0 for brevity)
        self.layer0_tensors = {
            "model.layers.0.self_attention.q_a_proj.weight": torch.randn(960, 6144, dtype=torch.bfloat16),
            "model.layers.0.self_attention.q_b_proj.weight": torch.randn(6144, 960, dtype=torch.bfloat16),
            "model.norm.weight": torch.randn(6144, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(151936, 6144, dtype=torch.bfloat16),
        }
        self.all_tensors = {**self.layer0_tensors, **self.layer78_tensors}

    def get_all_tensor_names(self) -> List[str]:
        return list(self.all_tensors.keys())

    def load_tensors(self, names: List[str]) -> Dict[str, torch.Tensor]:
        return {k: self.all_tensors[k] for k in names if k in self.all_tensors}

    def close(self):
        pass


class MockSaver:
    """Mock saver that collects tensors and writes a simple index."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.saved_tensors: Dict[str, torch.Tensor] = {}
        self.saved_types: Dict[str, str] = {}

    def add_tensors(self, tensors: Dict[str, torch.Tensor], tensor_types=None):
        self.saved_tensors.update(tensors)
        if tensor_types:
            self.saved_types.update(tensor_types)

    def finalize(self):
        # Write a simple index
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in self.saved_tensors.values())},
            "weight_map": {k: "mock-shard.safetensors" for k in self.saved_tensors},
        }
        with open(os.path.join(self.output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
        # Write quant_model_description
        with open(os.path.join(self.output_dir, "quant_model_description.json"), "w") as f:
            json.dump(self.saved_types, f, indent=2)


def run_backfill_test(test_name: str, quantize_mtp: bool, save_mtp_debug: bool):
    """Run a single backfill test with the given MTP flags."""
    print(f"\n{'='*60}")
    print(f"Test: {test_name}")
    print(f"  quantize_mtp={quantize_mtp}, save_mtp_debug={save_mtp_debug}")
    print(f"{'='*60}")

    tmpdir = tempfile.mkdtemp(prefix=f"mtp_test_{test_name}_")
    try:
        loader = MockChunkLoader(tmpdir)
        saver = MockSaver(tmpdir)

        all_original_keys = set(loader.get_all_tensor_names())
        touched_original_keys: Set[str] = set()
        mtp_names = ["model.layers.78"]

        # Simulate regular chunk processing (layer 0)
        for k in loader.layer0_tensors:
            touched_original_keys.add(k)
            saver.add_tensors(
                {k: loader.layer0_tensors[k]},
                tensor_types={k: "W4A16"},  # Simulated quantized
            )

        # Simulate MTP decision logic (from task.py)
        process_mtp = mtp_names and saver is not None and (quantize_mtp or save_mtp_debug)

        if process_mtp:
            print(f"  [MTP] process_mtp=True, loading and processing MTP chunk")
            mtp_chunk_tensors = loader.load_tensors(
                [k for k in all_original_keys if k.startswith("model.layers.78.")]
            )
            touched_original_keys.update(mtp_chunk_tensors.keys())

            if quantize_mtp:
                print(f"  [MTP] Quantizing MTP layer")
                tensor_types = {k: "W4A16" for k in mtp_chunk_tensors}
            else:
                print(f"  [MTP] Debug-only mode, saving as FLOAT")
                tensor_types = {k: "FLOAT" for k in mtp_chunk_tensors}

            saver.add_tensors(mtp_chunk_tensors, tensor_types=tensor_types)

        elif mtp_names and saver is not None:
            print(f"  [MTP] elif branch: will be saved as-is via backfill")
            # Do NOT mark as touched -- let backfill handle it

        # Simulate backfill logic (from task.py)
        missing_original_keys = sorted(all_original_keys - touched_original_keys)
        if missing_original_keys:
            layer_counts = {}
            for k in missing_original_keys:
                parts = k.split(".")
                if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers":
                    layer_key = f"layer {parts[2]}"
                else:
                    layer_key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
                layer_counts[layer_key] = layer_counts.get(layer_key, 0) + 1
            layer_summary = ", ".join(f"{k}={v}" for k, v in sorted(layer_counts.items()))
            print(f"  [Backfill] {len(missing_original_keys)} missing tensors ({layer_summary})")

            missing_tensors = loader.load_tensors(missing_original_keys)
            saver.add_tensors(
                missing_tensors,
                tensor_types={name: "FLOAT" for name in missing_original_keys},
            )
            print(f"  [Backfill] Saved {len(missing_tensors)} tensors as FLOAT")
        else:
            print(f"  [Backfill] No missing tensors")

        saver.finalize()

        # Verify results
        print(f"\n  --- Verification ---")
        saved = set(saver.saved_tensors.keys())
        layer78_saved = [k for k in saved if k.startswith("model.layers.78.")]
        layer0_saved = [k for k in saved if k.startswith("model.layers.0.")]
        norm_saved = [k for k in saved if "model.norm" in k]
        lm_head_saved = [k for k in saved if "lm_head" in k]

        print(f"  Layer 0: {len(layer0_saved)} tensors saved")
        print(f"  Layer 78: {len(layer78_saved)} tensors saved")
        print(f"  model.norm: {len(norm_saved)} tensors saved")
        print(f"  lm_head: {len(lm_head_saved)} tensors saved")

        # Check all source tensors are saved
        missing_from_output = all_original_keys - saved
        if missing_from_output:
            print(f"  FAIL: {len(missing_from_output)} tensors missing from output!")
            for k in sorted(missing_from_output)[:5]:
                print(f"    - {k}")
            return False

        # Check layer 78 tensor types
        layer78_types = set(saver.saved_types.get(k, "UNKNOWN") for k in layer78_saved)
        print(f"  Layer 78 types: {layer78_types}")

        if quantize_mtp:
            expected_type = "W4A16"
        else:
            expected_type = "FLOAT"

        if quantize_mtp and layer78_types != {"W4A16"}:
            print(f"  FAIL: Expected W4A16 for layer 78, got {layer78_types}")
            return False
        elif not quantize_mtp and layer78_types != {"FLOAT"}:
            print(f"  FAIL: Expected FLOAT for layer 78, got {layer78_types}")
            return False

        print(f"  PASS: All {len(all_original_keys)} tensors saved, types correct")
        return True

    finally:
        shutil.rmtree(tmpdir)


def main():
    print("MTP Backfill Logic Test")
    print("=" * 60)

    results = []

    # Case 1: quantize_mtp=False, save_mtp_debug=False (current GLM-5 config)
    results.append((
        "case1_no_mtp",
        run_backfill_test("case1_no_mtp", quantize_mtp=False, save_mtp_debug=False),
    ))

    # Case 2: quantize_mtp=True, save_mtp_debug=False
    results.append((
        "case2_quantize_mtp",
        run_backfill_test("case2_quantize_mtp", quantize_mtp=True, save_mtp_debug=False),
    ))

    # Case 3: quantize_mtp=False, save_mtp_debug=True
    results.append((
        "case3_debug_only",
        run_backfill_test("case3_debug_only", quantize_mtp=False, save_mtp_debug=True),
    ))

    # Case 4: quantize_mtp=True, save_mtp_debug=True
    results.append((
        "case4_both",
        run_backfill_test("case4_both", quantize_mtp=True, save_mtp_debug=True),
    ))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
