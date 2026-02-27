#!/usr/bin/env python
"""
Check QuIP quantized model contents.
"""

import os
import argparse
from safetensors.torch import load_file
import torch


def main():
    parser = argparse.ArgumentParser(description='Check QuIP model contents')
    parser.add_argument('model_path', type=str, nargs='?',
                        default='outputs/compressor/quip/opt-125m-w4-real',
                        help='Path to quantized model')
    args = parser.parse_args()

    print(f"Checking model at: {args.model_path}")
    print("=" * 60)

    # List files
    print("\nFiles:")
    for f in sorted(os.listdir(args.model_path)):
        fpath = os.path.join(args.model_path, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f}: {size:.2f} MB")

    # Load safetensors
    safetensors_path = os.path.join(args.model_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print("\nNo model.safetensors found!")
        return

    print(f"\nLoading {safetensors_path}...")
    state_dict = load_file(safetensors_path)

    # Categorize keys
    quip_keys = {}  # layer_name -> list of keys
    other_keys = []

    for key in state_dict.keys():
        if 'qweight' in key or 'scales' in key or 'scaleWH' in key or 'proj_seed' in key:
            layer_name = '.'.join(key.split('.')[:-1])
            if layer_name not in quip_keys:
                quip_keys[layer_name] = []
            quip_keys[layer_name].append(key)
        else:
            other_keys.append(key)

    print(f"\nQuIP Layers: {len(quip_keys)}")
    print("-" * 40)

    # Show first 3 layers as example
    for i, (layer_name, keys) in enumerate(list(quip_keys.items())[:3]):
        print(f"\n{layer_name}:")
        for k in sorted(keys):
            param_name = k.split('.')[-1]
            tensor = state_dict[k]
            dtype = tensor.dtype
            shape = tuple(tensor.shape)
            print(f"  {param_name}: shape={shape}, dtype={dtype}")

    if len(quip_keys) > 3:
        print(f"\n... and {len(quip_keys) - 3} more layers")

    # Summary stats
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total keys: {len(state_dict)}")
    print(f"  QuIP layers: {len(quip_keys)}")
    print(f"  Other keys: {len(other_keys)}")

    # Memory savings estimate
    if quip_keys:
        # Each QuIP layer has: qweight (4-bit packed), scales (fp16), scaleWH (fp32), seeds (int64)
        total_orig = 0  # Original fp32 weights
        total_quip = 0  # Quantized weights

        for layer_name, keys in quip_keys.items():
            for k in keys:
                tensor = state_dict[k]
                size = tensor.numel() * tensor.element_size()
                total_quip += size

                # Estimate original size (fp32 weights)
                if 'qweight' in k:
                    # qweight is [in/32*bits, out], so original is [out, in]
                    bits = 4
                    out_f = tensor.shape[1]
                    in_f = tensor.shape[0] * 32 // bits
                    total_orig += out_f * in_f * 4  # fp32

        print(f"\nEstimated memory:")
        print(f"  Original (fp32): {total_orig / 1024 / 1024:.2f} MB")
        print(f"  QuIP (4-bit): {total_quip / 1024 / 1024:.2f} MB")
        print(f"  Compression: {total_orig / total_quip:.2f}x")


if __name__ == '__main__':
    main()
