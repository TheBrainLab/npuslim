# INT8 Dynamic Quantization

INT8Dynamic is a post-training quantization algorithm that uses per-channel weight quantization with per-token activation quantization.

## Overview

- **Weight Quantization**: Per-channel, INT8
- **Activation Quantization**: Per-token, INT8
- **No Training Required**: Fast calibration-only approach
- **NPU-Native**: Optimized for Ascend vLLM-ascend deployment

## Configuration

```yaml
pipeline:
  - type: ptq
    algo_name: INT8Dynamic
    ignore_layers: []  # Optional: glob/regex patterns to skip
    algo_config:
      w_bits: 8              # Weight bit width
      weight: per-channel      # or per-tensor
      activation: per-token     # or per-tensor
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|--------|-------------|----------|
| `w_bits` | int | Weight bit width | 8 |
| `weight` | str | Weight quantization granularity | per-channel |
| `activation` | str | Activation quantization granularity | per-token |
| `ignore_layers` | list | Layer patterns to skip | [] |

## Example

```yaml
meta:
  type: llm

model:
  type: Qwen3
  model_path: /path/to/qwen3-0.6b

calib_dataset:
  dataset:
    type: C4Dataset
    num_samples: 256
    max_seq_length: 2048

pipeline:
  - type: ptq
    algo_name: INT8Dynamic
    algo_config:
      w_bits: 8
      weight: per-channel
      activation: per-token

  - type: save
    save_dir: ./outputs
```

## Run Command

```bash
python tools/run.py -c configs/opt/int8_dynamic/opt_125m-w8a8.yaml
```

## Deployment

The quantized model can be deployed directly with vLLM-ascend:

```bash
bash deploy/run_vllm.sh --model-path outputs/opt/int8_dynamic/opt_125m-w8a8 -d 4,5 -t 2 -q
```

The `-q` flag enables quantization support in vLLM.

## Notes

- LM head layers are skipped by default
- Use `ignore_layers` to exclude specific modules (e.g., `output.*`, `re:lm_head.*`)
- Calibration dataset quality affects final accuracy
