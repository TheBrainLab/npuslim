# NPUSlim

NPU-oriented model compression & quantization framework for large language models.

## Features

- Post-training quantization: INT8Dynamic, GPTQ, QuIP, SparseGPT
- Model support: Qwen3, OPT
- vLLM-ascend deployment integration
- Performance evaluation with lm-eval and evalscope

## Installation

```bash
pip install -e .
```

Requires CANN environment with `ASCEND_HOME_PATH` set.

## Quick Start

### Quantization

```bash
# Use mirror if HuggingFace is inaccessible
export HF_ENDPOINT="https://hf-mirror.com"

# GPU
python tools/run.py -c configs/compressor/int8_dyn/qwen3/qwen3_0_6b.yaml

# NPU
python tools/run.py -c configs/compressor/int8_dyn/qwen3/ascend-qwen3_0_6b.yaml
```

### Deployment (vLLM)

```bash
# GPU
bash tools/serve/deploy_vllm.sh outputs/compressor/int8_dyn/qwen3/qwen3_0_6b -d 0 -t 1

# NPU
bash tools/serve/deploy_vllm.sh outputs/compressor/int8_dyn/qwen3/ascend-qwen3_0_6b -d 0 -t 1 -q
```

### Evaluation

**LM-Eval Harness** (supports 3 backends: `vllm`, `hf`, `api`):

```bash
# vLLM backend (fastest, direct loading - no server needed)
bash tools/eval/run_lmeval.sh outputs/model --backend vllm --tasks wikitext -d 0

# HuggingFace backend
bash tools/eval/run_lmeval.sh outputs/model --backend hf --tasks wikitext -d 0

# API backend (requires running server)
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1
bash tools/eval/run_lmeval.sh outputs/model --backend api --tasks wikitext
```

**Stress Test** (requires running vLLM server):

```bash
# Step 1: Deploy vLLM server first
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# Step 2: Run stress test against running server
bash tools/eval/run_stress_test.sh outputs/model
```

## Tool Scripts

| Script | Description |
|-------|-------------|
| `tools/serve/deploy_vllm.sh` | Deploy vLLM inference server |
| `tools/eval/run_lmeval.sh` | Run lm-evaluation-harness (backends: vllm, hf, api) |
| `tools/eval/run_stress_test.sh` | Run stress test via API (requires running server) |

### Common Options

Server deployment options:
- `-d, --devices` - Device IDs (e.g., `0,1` or `4,5`)
- `-t, --tp` - Tensor parallel size
- `--gpu-memory` - GPU memory utilization (default: 0.8)
- `--max-model-len` - Max model length (default: 4096)
- `-q, --quantization` - Quantization method (auto-detected on NPU)

LM-Eval options:
- `--backend` - Backend type: `vllm`, `hf`, or `api` (default: vllm)
- `--tasks` - Comma-separated benchmark tasks (default: wikitext)
- `--limit` - Limit number of samples per task
- `--log-samples` - Save model outputs for debugging

Use `--help` to see all options for each script.

## Configuration

Edit config files in `configs/compressor/<algo>/<model>/` to customize model path, quantization parameters, and pipeline tasks.

```yaml
model:
  model_path: your/model/path

pipeline:
  - type: ptq
    algo_name: INT8Dynamic
```

## Architecture

- `src/npuslim/slim_engine.py` - Orchestrator managing resources and task pipeline
- `src/npuslim/utils/factory.py` - Factory pattern for models, datasets, tasks, compressors
- `src/npuslim/compressor/quantizer/` - Quantization algorithms (INT8Dynamic, GPTQ, QuIP, SparseGPT)
- `src/npuslim/vllm_plugin/` - vLLM-ascend integration
- `tools/utils/common.sh` - Shared bash utilities (logging, device detection)

## License

Apache-2.0
