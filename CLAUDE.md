# NPUSlim Framework

## Run Commands

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
```bash
# LM-Eval in GPU (wikitext, ceval, etc.)
bash tools/eval/run_lmeval.sh outputs/compressor/int8_dyn/qwen3/qwen3_0_6b --tasks wikitext -d 0 -t 1

# NPU
bash tools/eval/run_lmeval.sh outputs/compressor/int8_dyn/qwen3/ascend-qwen3_0_6b --tasks wikitext -d 0 -t 1 -q

# Stress test with evalscope (full pipeline: deploy → benchmark → cleanup)
bash tools/eval/run_stress_test.sh outputs/compressor/int8_dyn/qwen3/ascend-qwen3_0_6b -d 0 -t 1
```

## Architecture

### Core Classes

**SlimEngine** (`src/npuslim/slim_engine.py`):
- Orchestrator managing global resources (models, dataloader, tokenizer)
- Builds/executes task pipeline from config
- Resources dict: `main_model`, `draft_model`, `student_model`, `dataloader`, `tokenizer`, `engine`

**Factory Pattern** (`src/npuslim/utils/factory.py`):
- `ModelFactory.create(config=model_cfg)` - Creates Qwen3, OPT models
- `DatasetFactory.create(config=dataset_cfg)` - Creates C4, WikiText2, MMLU datasets
- `TaskFactory.create(task_key, raw_config, resources)` - Creates ptq, eval, save tasks
- `CompressorFactory.create(algo_name, ...)` - Creates quantization algorithms
- `SaverFactory.create(format_name, ...)` - Creates HuggingFace/Ascend savers

**Base Classes**:
- `BaseLLMModel`: Model wrapper with HF/ModelScope hub support via `get_hub_class()`
- `BaseTask`: Abstract task with `ConfigClass` auto-parsing and `_resolve_layer_names()` for glob/regex patterns
- `BaseCompressorAlgo`: Unified interface with `prepare()`, `calibrate()`, `convert()`, `compress()`, `apply_masks()`
- `BaseObserver`: Activation/weight observers (AbsMaxActivation, AbsMaxWeight, PTQObserver)
- `BaseSaver`: Model serialization base class

### Pipeline Task Types

- `ptq`: Post-training quantization
- `eval`: Model evaluation (perplexity, accuracy)
- `save`: Export quantized model

### Directory Structure

```
src/npuslim/
├── slim_engine.py          # Main orchestrator
├── cli/                    # Command-line interface
├── cann_ops/               # CANN-specific operators
│   ├── quant/              # Quantization primitives (PTQ, QAT tools)
│   ├── llm_ptq/            # LLM-specific PTQ utilities
│   ├── sparse/             # Sparsity utilities
│   ├── lowbit/             # Low-bit quantization
│   └── multi_modal/        # Multi-modal support
├── compressor/
│   ├── quantizer/          # Quantization algorithms
│   │   ├── int8_dyn/       # INT8 dynamic quantization
│   │   ├── gptq/           # GPTQ algorithm
│   │   ├── quip/           # QuIP algorithm
│   │   └── sparsegpt/      # SparseGPT pruning + quantization
│   ├── observers/          # Activation/weight observers
│   └── core/               # Quantization utilities, LayerWiseScheduler
├── dataset/                # Calibration datasets (C4, WikiText2, MMLU)
├── tasks/                  # Pipeline tasks (ptq, eval, save)
├── saver/                  # Model savers (HuggingFace, Ascend)
├── plugins/                # Integration plugins
│   ├── vllm/               # vLLM model executor plugins
│   └── vllm_ascend/        # vLLM-Ascend quantization methods
└── utils/                  # Config parser, backend utilities, factory

tools/
├── run.py                  # Main entry point
├── serve/deploy_vllm.sh    # vLLM server deployment
├── eval/                   # Evaluation scripts
│   ├── run_lmeval.sh       # LM-Eval harness
│   └── run_stress_test.sh  # Stress test pipeline
└── utils/common.sh         # Shared bash utilities
```

## Config Format

```yaml
meta:
  type: llm
  work_dir: ./logs

model:
  type: Qwen3  # or OPT
  model_path: Qwen/Qwen3-0.6B
  model_hub: hf  # or ms (ModelScope)
  model_kwargs:
    trust_remote_code: true
    low_cpu_mem_usage: true

calib_dataset:
  dataset:
    type: C4Dataset  # or WikiText2, MMLU, TextDataset
    num_samples: 256
    max_seq_length: 2048
  dataloader:
    batch_size: 1

pipeline:
  - type: ptq
    algo_name: INT8Dynamic  # or GPTQ, QuIP, SparseGPT
    ignore_layers: []  # glob/regex patterns
    algo_config:
      w_bits: 8

  - type: eval

  - type: save
    save_dir: ./outputs
    format: AscendSaver  # or HuggingFaceSaver
```

## Quantization Algorithms

| Algorithm | Directory | Description |
|-----------|-----------|-------------|
| INT8Dynamic | `int8_dyn/` | Per-channel weight, per-token activation |
| GPTQ | `gptq/` | Activation-aware weight quantization |
| QuIP | `quip/` | Quaternion-inspired with vector balancing |
| SparseGPT | `sparsegpt/` | Structured pruning + quantization |

## Plugin System

Entry points defined in `pyproject.toml`:
```toml
[project.entry-points."vllm.general_plugins"]
npuslim = "npuslim.plugins:register"

[project.entry-points."transformers.quantizers"]
quip = "npuslim.plugins.transformers.quantizers.quantizer_quip:QuipHfQuantizer"
```

### Plugin Architecture

`npuslim.plugins:register()` is the main entry point that:
1. Calls `plugins.vllm.register()` - Core vLLM patches
2. Calls `plugins.transformers.register()` - HuggingFace quantizers
3. Conditionally calls `plugins.vllm_ascend.register()` - NPU-specific patches

### File Structure Convention

**Plugin paths MUST mirror the target framework's structure:**
```
src/npuslim/plugins/
├── vllm/                              # Patches vllm.* modules
│   └── model_executor/models/
│       └── qwen3_moe.py               # Patches vllm.model_executor.models.qwen3_moe
├── vllm_ascend/                       # Patches vllm_ascend.* modules
│   └── quantization/
│       ├── method_adapters.py         # Patches vllm_ascend.quantization.method_adapters
│       └── methods/w4a16_linear.py    # New quantization scheme for vLLM-Ascend
└── transformers/                      # Patches transformers.* modules
    └── quantizers/quantizer_quip.py   # Registers QuipHfQuantizer
```

### Patch Mechanism

Use `@register_patch(target_module)` decorator from `npuslim.plugins.registry`:
```python
from npuslim.plugins.registry import register_patch

@register_patch("vllm_ascend.quantization.method_adapters")
def patch_process_weight(module):
    original = module.AscendLinearMethod.process_weight
    module.AscendLinearMethod.process_weight = patched_version
```

For vLLM-Ascend quantization schemes, use their `@register_scheme` decorator:
```python
from vllm_ascend.quantization.methods import register_scheme

@register_scheme("W4A16", "linear")
class MyW4A16LinearMethod(AscendLinearScheme):
    ...
```

## Saver Modules

- **HuggingFaceSaver**: Standard HF format export
- **AscendSaver**: Generates `quant_model_description.json` for vLLM-Ascend

## Requirements

- **CANN**: Set `ASCEND_HOME_PATH` env var before installation
- **Python**: >=3.11
