# Getting Started

This guide will walk you through installing NPUSlim and performing your first model quantization.

## Installation

### Prerequisites

- Python >= 3.11
- CANN toolkit installed
- `ASCEND_HOME_PATH` environment variable set

```bash
# Check CANN installation
echo $ASCEND_HOME_PATH
```

### Install NPUSlim

```bash
cd /path/to/npuslim
pip install -e . -v
```

The `-e` flag installs in development mode, and `-v` enables verbose output.

## Your First Quantization

### 1. Prepare Config

Edit a config file in `configs/compressor/`, for example `int8_dyn/qwen3_0_6b.yaml`:

```yaml
meta:
  type: llm
  work_dir: ./logs

model:
  type: Qwen3
  model_path: Qwen/Qwen3-0.6B
  model_hub: hf
  model_kwargs:
    trust_remote_code: true
    low_cpu_mem_usage: true

calib_dataset:
  dataset:
    type: C4Dataset
    num_samples: 256
    max_seq_length: 2048
  dataloader:
    batch_size: 1

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

### 2. Run Quantization

```bash
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml
```

### 3. Check Output

After completion, the quantized model will be saved to `./outputs/<model_name>/`.

```
outputs/qwen3_int8_dyn/
├── config.json
├── tokenizer.json
└── model.safetensors
```

## Next Steps

- [Deploy with vLLM](vllm_deployment.md)
- [Try different quantization algorithms](int8_dyn.md)
- [Evaluate model accuracy](../benchmark/index.md)
