from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from npuslim.algorithms.quantization.sparsegpt.sparsegpt_algo import _pack_sparse24
from npuslim.plugins.vllm.model_executor.layers._stacked_sparse24 import (
    load_stacked_sparse24_weight_index,
)

_K_TILE_INDEX = 8


def _make_sparse24_weight(rows: int, cols: int, seed: int) -> torch.Tensor:
    assert cols % 4 == 0
    g = torch.Generator().manual_seed(seed)
    weight = torch.zeros(rows, cols, dtype=torch.int8)
    values = torch.randint(1, 8, (rows, cols // 4, 2), generator=g, dtype=torch.int64)
    positions = torch.randint(0, 6, (rows, cols // 4), generator=g, dtype=torch.int64)
    pos_map = torch.tensor(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        dtype=torch.int64,
    )
    chosen = pos_map[positions]
    signs = torch.randint(0, 2, values.shape, generator=g, dtype=torch.int64) * 2 - 1
    values = (values * signs).to(torch.int8)
    grouped = weight.view(rows, cols // 4, 4)
    grouped.scatter_(2, chosen[:, :, :1], values[:, :, :1])
    grouped.scatter_(2, chosen[:, :, 1:], values[:, :, 1:])
    return weight


def _decode_sparse24_to_dense(weight: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    n = weight.shape[0]
    k = weight.shape[1] * 2
    pad_n = math.ceil(n / 16) * 16
    k8 = k // 8
    num_tiles = index.numel() // (pad_n * _K_TILE_INDEX)

    idx_mat = torch.zeros((pad_n, num_tiles * _K_TILE_INDEX), dtype=torch.uint8)
    offset = 0
    for tile in range(num_tiles):
        span = pad_n * _K_TILE_INDEX
        idx_mat[:, tile * _K_TILE_INDEX:(tile + 1) * _K_TILE_INDEX] = index[
            offset:offset + span
        ].reshape(pad_n, _K_TILE_INDEX)
        offset += span
    idx_mat = idx_mat[:n, :k8]

    byte_vals = idx_mat.reshape(n, -1)
    idx0 = (byte_vals & 0b11).to(torch.int64)
    idx1 = ((byte_vals >> 2) & 0b11).to(torch.int64)
    idx2 = ((byte_vals >> 4) & 0b11).to(torch.int64)
    idx3 = ((byte_vals >> 6) & 0b11).to(torch.int64)
    pair_vals = torch.stack([idx0, idx1, idx2, idx3], dim=-1).reshape(n, -1)
    pair_vals = pair_vals.reshape(n, k // 4, 2)

    dense = torch.zeros((n, k), dtype=torch.int8)
    dense_vals = weight.reshape(n, k // 4, 2)
    first_pos = pair_vals[:, :, 0]
    second_pos = pair_vals[:, :, 1] + 1
    for pos in range(4):
        mask = first_pos == pos
        dense[:, pos::4][mask] = dense_vals[:, :, 0][mask]
    for pos in range(1, 4):
        mask = second_pos == pos
        dense[:, pos::4][mask] = dense_vals[:, :, 1][mask]
    return dense


def _naive_stack(target_len: int, shards: list[tuple[int, torch.Tensor]]) -> torch.Tensor:
    target = torch.zeros(target_len, dtype=torch.uint8)
    offset = 0
    for shard_rows, shard_index in shards:
        target[offset:offset + shard_rows] = shard_index[:shard_rows]
        offset += shard_rows
    return target


def test_qkv_sparse24_stacked_index_merge_matches_reference():
    q = _make_sparse24_weight(32, 32, seed=1)
    k = _make_sparse24_weight(16, 32, seed=2)
    v = _make_sparse24_weight(16, 32, seed=3)
    q_packed, q_index = _pack_sparse24(q)
    k_packed, k_index = _pack_sparse24(k)
    v_packed, v_index = _pack_sparse24(v)

    fused_weight = torch.cat([q_packed, k_packed, v_packed], dim=0)
    fused_index = torch.zeros(64 * _K_TILE_INDEX, dtype=torch.uint8)
    layer = SimpleNamespace(num_heads=4, head_size=8, num_kv_heads=2, v_head_size=8)
    param = SimpleNamespace(data=fused_index)

    load_stacked_sparse24_weight_index(layer, param, q_index, "q")
    load_stacked_sparse24_weight_index(layer, param, k_index, "k")
    load_stacked_sparse24_weight_index(layer, param, v_index, "v")

    dense = _decode_sparse24_to_dense(fused_weight, fused_index)
    ref = torch.cat([q, k, v], dim=0)
    assert torch.equal(dense, ref)


def test_merged_sparse24_stacked_index_merge_matches_reference():
    gate = _make_sparse24_weight(48, 32, seed=4)
    up = _make_sparse24_weight(48, 32, seed=5)
    gate_packed, gate_index = _pack_sparse24(gate)
    up_packed, up_index = _pack_sparse24(up)

    fused_weight = torch.cat([gate_packed, up_packed], dim=0)
    fused_index = torch.zeros(96 * _K_TILE_INDEX, dtype=torch.uint8)
    layer = SimpleNamespace(output_sizes=[48, 48])
    param = SimpleNamespace(data=fused_index)

    load_stacked_sparse24_weight_index(layer, param, gate_index, 0)
    load_stacked_sparse24_weight_index(layer, param, up_index, 1)

    dense = _decode_sparse24_to_dense(fused_weight, fused_index)
    ref = torch.cat([gate, up], dim=0)
    assert torch.equal(dense, ref)


def test_naive_sparse24_stacked_index_copy_is_incorrect():
    q = _make_sparse24_weight(32, 32, seed=6)
    k = _make_sparse24_weight(16, 32, seed=7)
    v = _make_sparse24_weight(16, 32, seed=8)
    q_packed, q_index = _pack_sparse24(q)
    k_packed, k_index = _pack_sparse24(k)
    v_packed, v_index = _pack_sparse24(v)

    fused_weight = torch.cat([q_packed, k_packed, v_packed], dim=0)
    naive_index = _naive_stack(
        64 * _K_TILE_INDEX,
        [(q.shape[0], q_index), (k.shape[0], k_index), (v.shape[0], v_index)],
    )

    dense = _decode_sparse24_to_dense(fused_weight, naive_index)
    ref = torch.cat([q, k, v], dim=0)
    assert not torch.equal(dense, ref)
