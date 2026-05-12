# NPUSlim Framework

## Run Commands

### Quantization
```bash
# Use mirror if HuggingFace is inaccessible
export HF_ENDPOINT="https://hf-mirror.com"

# GPU (INT8)
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml

# GPU (GPTQ)
python tools/run.py -c configs/opt/gptq/opt_125m-w4a16.yaml

# NPU (use device_map: npu in config)
python tools/run.py -c configs/qwen3/int8_dynamic/qwen3_8b-w8a8.yaml
```

### Deployment (vLLM)
```bash
# GPU
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1

# NPU
bash tools/serve/deploy_vllm.sh outputs/model -d 0 -t 1 -q
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

## Architecture

### Core Design

NPUSlim v2 uses a **streaming-first, chunk-based quantization pipeline**. Instead of loading the full model into memory, it streams through checkpoint shards chunk-by-chunk via `ChunkLoader`, applies algorithms, and writes results incrementally via `StreamingHuggingFaceSaver`.

### Core Module (`src/npuslim/core/`)

**SlimEngine** (`core/engine.py`):
- Pipeline orchestrator: creates `ResourceManager` from config resources, builds and executes recipe tasks
- Each task receives `resource_manager` for lazy resource acquisition

**Registry Pattern** (`core/factory.py`):
- Single `Registry` class with `register()`, `register_lazy()`, `get()`, `create()`, `list()` methods
- 5 singleton registries: `AlgorithmRegistry`, `ModelRegistry`, `DatasetRegistry`, `TaskRegistry`, `SaverRegistry`
- Lazy loading: modules are imported only on first `get()`/`create()` call

**ResourceManager** (`core/resource_manager.py`):
- Resolves `@resource_id` references from config
- Lazy model/dataset instantiation via registry lookup

**BackendHandler** (`core/backend.py`):
- Unified CPU/CUDA/NPU backend abstraction

**bootstrap_from_path()** (`core/bootstrap.py`):
- CLI bootstrap: YAML loading, config parse/validate, logging setup

### Base Classes

- **BaseAlgorithm** (`algorithms/base_algo.py`): `process_chunk(chunk: ChunkContext) -> ChunkContext` with `on_start()`/`on_finish()` lifecycle hooks
- **BaseQuantizationAlgorithm** (`algorithms/quantization/base_quant_algo.py`): Adds `set_runtime_context()`, `should_skip_name()` for glob/regex skip matching, `_mark_model_quantized()`
- **BaseLLMModel** (`models/base_model.py`): Model wrapper with HF/ModelScope hub support, `prepare_empty_model()` for meta-device skeleton
- **BaseTask** (`tasks/base_task.py`): Lifecycle hooks (`on_start`, `on_finish`, `execute`) with lazy resource acquisition via `ResourceManager`
- **BaseSaver** (`savers/base_saver.py`): Streaming tensor writer interface
- **BaseDataset** (`datasets/base_dataset.py`): Dataset with `processor` argument, `collate_fn` static method

### Task Types

- **compressor** (`tasks/compressor/`): Streaming quantization task — handles chunk loading, algorithm invocation, skip-layer resolution, and saver coordination all in one unified task

### Directory Structure

```
src/npuslim/
├── core/                   # Core framework
│   ├── engine.py           # SlimEngine orchestrator
│   ├── factory.py          # Registry pattern (5 singleton registries)
│   ├── resource_manager.py # Lazy resource acquisition
│   ├── backend.py          # CPU/CUDA/NPU backend handler
│   └── bootstrap.py        # CLI bootstrap
├── config/                 # Config parsing
│   ├── schema.py           # Dataclasses (EngineConfig, ResourceConfig, etc.)
│   ├── parser.py           # YAML/dict -> EngineConfig
│   ├── validator.py        # Reference checking
│   └── printer.py          # Pretty-printing, logging setup
├── algorithms/             # Quantization algorithms
│   ├── base_algo.py        # BaseAlgorithm (process_chunk interface)
│   └── quantization/
│       ├── base_quant_algo.py  # Shared runtime context + skip-matching
│       ├── gptq/           # GPTQ algorithm
│       └── int8_dynamic/   # INT8 dynamic quantization
├── models/                 # Model wrappers
│   ├── base_model.py       # BaseLLMModel
│   ├── qwen3/              # Qwen3 model
│   ├── opt/                # OPT model
│   └── glm5/               # GLM-5 model (GlmMoeDsa)
├── datasets/               # Calibration datasets
│   ├── base_dataset.py     # BaseDataset
│   ├── c4_dataset.py       # C4 dataset (streaming + local cache)
│   └── text_dataset.py     # Text dataset (JSONL + Parquet)
├── tasks/                  # Pipeline tasks
│   ├── base_task.py        # BaseTask with lifecycle hooks
│   └── compressor/         # Streaming compressor task
│       ├── context.py      # ChunkContext, LayerInfo, ModuleInfo
│       ├── loader.py       # ChunkLoader (streaming safetensors reader)
│       └── task.py         # CompressorTask
├── savers/                 # Model savers
│   ├── base_saver.py       # BaseSaver interface
│   └── hf_saver.py         # StreamingHuggingFaceSaver (safetensors)
├── hooks/                  # Lifecycle hook system (HookType, HookRegistry)
├── distributed/            # Distributed execution support
├── cann_ops/               # CANN-specific operators
│   ├── quant/              # Quantization primitives (PTQ, QAT tools)
│   ├── llm_ptq/            # LLM-specific PTQ utilities
│   ├── sparse/             # Sparsity utilities
│   ├── lowbit/             # Low-bit quantization
│   └── multi_modal/        # Multi-modal support
├── plugins/                # Integration plugins
│   ├── vllm/               # vLLM model executor plugins
│   │   └── model_executor/models/
│   │       ├── qwen3_moe.py
│   │       └── kimi_k2_mcore.py
│   ├── vllm_ascend/        # vLLM-Ascend quantization methods
│   └── transformers/       # HuggingFace quantizers
└── cli/                    # CLI entry point (tools/run.py)

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
metadata:
  name: "Qwen3_INT8_Recipe"
  description: "INT8 quantization for Qwen3"

resources:
  - id: qwen3
    type: Qwen3
    path: Qwen/Qwen3-0.6B
    model_hub: hf          # or ms (ModelScope)
    device_map: cuda       # or cpu, npu

  - id: calib_data
    type: C4
    num_samples: 128
    max_seq_length: 2048

recipe:
  - name: "INT8_Quantization"
    type: compressor
    model: "@qwen3"                         # Reference to resource
    dataloader:
      dataset: "@calib_data"               # Reference to resource
      batch_size: 1
    algorithm:
      type: INT8Dynamic                    # or GPTQ
      wbits: 8
    ignore_layers: []                      # glob/regex patterns
    execution:
      mode: streaming
      chunk_size: 4
    saver:
      type: StreamingHuggingFaceSaver
      save_dir: "./outputs"
```

### Config Schema

- **EngineConfig**: `metadata` + `resources` (list) + `recipe` (list of tasks)
- **ResourceConfig**: `id`, `type`, extra fields passed to constructor
- **RecipeTaskConfig**: `name`, `type`, `model` (@ref), `dataloader`, `algorithm`, `saver`, `execution`
- Resource references use `@id` syntax (e.g., `@qwen3`, `@calib_data`)

## Quantization Algorithms

| Algorithm | Registry Name | Description |
|-----------|--------------|-------------|
| INT8Dynamic | `INT8Dynamic` | Per-channel weight, per-token activation quantization |
| GPTQ | `GPTQ` | Activation-aware weight quantization with Hessian statistics |

## Supported Models

| Model | Registry Name | Description |
|-------|--------------|-------------|
| Qwen3 | `Qwen3` (or `Qwen3Model`) | Qwen3 series |
| OPT | `OPT` (or `OPTModel`) | Meta OPT series |
| GLM-5 | `GLM5` | GlmMoeDsa with MLA attention |

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
│       ├── qwen3_moe.py               # Patches vllm.model_executor.models.qwen3_moe
│       └── kimi_k2_mcore.py           # Patches vllm.model_executor.models.kimi_k2_mcore
├── vllm_ascend/                       # Patches vllm_ascend.* modules
│   └── quantization/
│       ├── method_adapters.py         # Patches vllm_ascend.quantization.method_adapters
│       └── methods/w4a16_linear.py    # Quantization scheme for vLLM-Ascend
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

- **StreamingHuggingFaceSaver**: Streaming safetensors writer with auto-flush on size threshold. Writes `model.safetensors.index.json`, copies auxiliary files, saves config/tokenizer. Generates `quant_model_description.json` for Ascend runtime when NPU mode is detected.

## Requirements

- **CANN**: Set `ASCEND_HOME_PATH` env var before installation
- **Python**: >=3.11
