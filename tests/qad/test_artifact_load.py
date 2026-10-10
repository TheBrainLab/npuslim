"""B2 frozen-source artifact loader: int4 unpack + dequant + rank-slice scatter unit tests.

Covers the W4A8 deploy-artifact dequantization pipeline added to
workspace/glm52-qad/stream_load_checkpoint.py:
  - unpack_int4: row-pair nibble packing (even=low, odd=high, two's complement [-8,7])
  - _dequant_artifact_weight: int4-packed / int8 / float-passthrough branches
  - _scatter_rank_slice: ZeRO-3 flat-slice overlap copy

Runs on CPU (no NPU, no artifact, no deepspeed needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

for _p in [
    Path(__file__).resolve().parents[2] / "workspace" / "glm52-qad",
    Path("/llm_infer_workspace/ylt/glm52-qad"),
]:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

from stream_load_checkpoint import (  # noqa: E402
    unpack_int4,
    _dequant_artifact_weight,
    _scatter_rank_slice,
)


def _pack_int4(signed: torch.Tensor) -> torch.Tensor:
    """Inverse of unpack_int4: [out, in] int32 [-8,7] -> [out//2, in] int8."""
    u = (signed & 0x0F).to(torch.uint8)
    packed = (u[1::2] << 4) | u[0::2]
    return packed.to(torch.int8)


class TestUnpackInt4:
    def test_roundtrip_known_values(self):
        signed = torch.tensor([[-8, 7], [-1, 0], [3, -4], [5, 5]], dtype=torch.int32)
        packed = _pack_int4(signed)
        out = unpack_int4(packed)
        assert out.shape == (4, 2)
        assert torch.equal(out, signed)

    def test_twos_complement_negatives(self):
        signed = torch.tensor([[-8, -7, -1], [-8, -1, -2]], dtype=torch.int32)
        packed = _pack_int4(signed)
        out = unpack_int4(packed)
        assert torch.equal(out, signed)

    def test_even_odd_nibble_order(self):
        signed = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32)
        packed = _pack_int4(signed)
        u8 = (packed.to(torch.int32) & 0xFF).to(torch.uint8)
        low = (u8[0] & 0x0F)[0].item()
        high = ((u8[0] >> 4) & 0x0F)[0].item()
        assert low == 1
        assert high == 2

    def test_full_range(self):
        torch.manual_seed(0)
        signed = torch.randint(-8, 8, (100, 50), dtype=torch.int32)
        packed = _pack_int4(signed)
        out = unpack_int4(packed)
        assert torch.equal(out, signed)


class TestDequantArtifactWeight:
    def test_int4_packed_dequant(self):
        torch.manual_seed(1)
        out_rows, in_cols = 8, 4
        q_int = torch.randint(-8, 8, (out_rows, in_cols), dtype=torch.int32)
        scale = torch.rand(out_rows, 1) * 0.1 + 0.01
        offset = torch.randn(out_rows, 1) * 0.05
        packed = _pack_int4(q_int)
        assert packed.dtype == torch.int8 and packed.shape == (out_rows // 2, in_cols)

        dequant = _dequant_artifact_weight(packed, scale, offset)
        ref = q_int.float() * scale + offset
        assert dequant.shape == (out_rows, in_cols)
        assert torch.allclose(dequant, ref, atol=1e-6)

    def test_float_passthrough(self):
        w = torch.randn(4, 3, dtype=torch.bfloat16)
        out = _dequant_artifact_weight(w, torch.empty(0), torch.empty(0))
        assert torch.equal(out, w)

    def test_int8_genuine_dequant(self):
        out_rows, in_cols = 6, 4
        w = torch.randint(-128, 128, (out_rows, in_cols), dtype=torch.int8)
        scale = torch.rand(out_rows, 1) * 0.01
        offset = torch.zeros(out_rows, 1)
        dequant = _dequant_artifact_weight(w, scale, offset)
        ref = w.to(torch.int32).float() * scale + offset
        assert torch.allclose(dequant, ref, atol=1e-6)

    def test_scale_shape_mismatch_raises(self):
        w = torch.zeros(4, 3, dtype=torch.int8)
        scale = torch.zeros(5, 1)
        offset = torch.zeros(5, 1)
        try:
            _dequant_artifact_weight(w, scale, offset)
            assert False, "should have raised"
        except ValueError:
            pass


class TestScatterRankSlice:
    def test_full_overlap(self):
        ds_tensor = torch.zeros(10, dtype=torch.bfloat16)
        info = {"ds_tensor": ds_tensor, "rank_start": 0, "rank_end": 10, "param": None}
        src = torch.arange(10, dtype=torch.bfloat16)
        _scatter_rank_slice(info, src, flat_offset=0)
        assert torch.equal(ds_tensor, src)

    def test_partial_overlap(self):
        ds_tensor = torch.zeros(10, dtype=torch.bfloat16)
        info = {"ds_tensor": ds_tensor, "rank_start": 5, "rank_end": 10, "param": None}
        src = torch.arange(20, dtype=torch.bfloat16)
        _scatter_rank_slice(info, src, flat_offset=0)
        assert torch.equal(ds_tensor[:5], src[5:10])
        assert ds_tensor[5:].sum() == 0

    def test_no_overlap(self):
        ds_tensor = torch.zeros(10, dtype=torch.bfloat16)
        info = {"ds_tensor": ds_tensor, "rank_start": 0, "rank_end": 5, "param": None}
        src = torch.arange(10, dtype=torch.bfloat16)
        _scatter_rank_slice(info, src, flat_offset=5)
        assert torch.equal(ds_tensor, torch.zeros(10, dtype=torch.bfloat16))

    def test_fused_offset(self):
        ds_tensor = torch.zeros(20, dtype=torch.bfloat16)
        info = {"ds_tensor": ds_tensor, "rank_start": 0, "rank_end": 20, "param": None}
        src = torch.tensor([1, 2, 3, 4], dtype=torch.bfloat16)
        _scatter_rank_slice(info, src, flat_offset=10)
        assert torch.equal(ds_tensor[10:14], src)
        assert ds_tensor[:10].sum() == 0 and ds_tensor[14:].sum() == 0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
