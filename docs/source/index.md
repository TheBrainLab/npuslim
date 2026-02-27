# NPUSlim

NPU-oriented model compression & quantization framework for large language models.

NPUSlim is designed specifically for Huawei Ascend NPU, providing efficient post-training quantization (PTQ) algorithms that seamlessly integrate with vLLM-ascend for deployment.

```{toctree}
:maxdepth: 2
:hidden:

Tutorials <tutorials/index>
API Reference <reference/index>
Performance Benchmarks <benchmark/index>
FAQ <faq/index>
About <about>
```

## Features

- **Multiple PTQ Algorithms**: INT8Dynamic, GPTQ, QuIP, SparseGPT
- **NPU-Optimized**: Designed for Ascend NPU with vLLM-ascend integration
- **Flexible Pipeline**: Modular task system for quantization, evaluation, and deployment
- **Easy Configuration**: YAML-based configs with factory pattern for extensibility

## Quick Start

### Installation

```bash
pip install -e . -v
```

Requires CANN environment with `ASCEND_HOME_PATH` set.

### Quantize a Model

```bash
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml
```

### Deploy with vLLM

```bash
bash deploy/run_vllm.sh --model-path outputs/qwen3_int8_dyn -d 4,5 -t 2 -q
```

## Supported Models

- Qwen3
- OPT

More models can be added by implementing the `BaseLLMModel` interface.

## License

Apache-2.0
