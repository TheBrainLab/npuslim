from __future__ import annotations

import math

STACKED_WEIGHT_LOADER_ATTR = "stacked_weight_loader"
_K_TILE_INDEX = 8


def _qkv_shard_meta(layer, shard_id: str) -> tuple[int, int]:
    if shard_id == "q":
        return 0, layer.num_heads * layer.head_size
    if shard_id == "k":
        return layer.num_heads * layer.head_size, layer.num_kv_heads * layer.head_size
    if shard_id == "v":
        offset = (layer.num_heads + layer.num_kv_heads) * layer.head_size
        return offset, layer.num_kv_heads * layer.v_head_size
    raise ValueError(f"Unsupported QKV shard id: {shard_id!r}")


def _merged_shard_meta(layer, shard_id: int) -> tuple[int, int]:
    output_sizes = getattr(layer, "output_sizes", None)
    if output_sizes is None:
        raise AttributeError("Merged layer is missing output_sizes")
    shard_id = int(shard_id)
    shard_size = int(output_sizes[shard_id])
    shard_offset = sum(int(size) for size in output_sizes[:shard_id])
    return shard_offset, shard_size


def load_stacked_sparse24_weight_index(
    layer,
    param,
    loaded_weight,
    shard_id: str | int,
) -> None:
    """Merge a standalone Sparse24 tiled index into a fused stacked buffer."""

    if isinstance(shard_id, str):
        row_offset, row_count = _qkv_shard_meta(layer, shard_id)
    else:
        row_offset, row_count = _merged_shard_meta(layer, shard_id)

    shard_pad = math.ceil(row_count / 16) * 16
    if loaded_weight.numel() % (shard_pad * _K_TILE_INDEX) != 0:
        raise ValueError(
            "Sparse24 shard index has an unexpected size: "
            f"shape={tuple(loaded_weight.shape)}, row_count={row_count}"
        )

    num_tiles = loaded_weight.numel() // (shard_pad * _K_TILE_INDEX)
    target = param.data
    if target.numel() % (num_tiles * _K_TILE_INDEX) != 0:
        raise ValueError(
            "Sparse24 fused index has an unexpected size: "
            f"shape={tuple(target.shape)}, num_tiles={num_tiles}"
        )

    fused_pad = target.numel() // (num_tiles * _K_TILE_INDEX)
    target_view = target.view(num_tiles, fused_pad, _K_TILE_INDEX)
    source_view = loaded_weight.view(num_tiles, shard_pad, _K_TILE_INDEX)
    target_view[:, row_offset:row_offset + row_count, :].copy_(
        source_view[:, :row_count, :]
    )
