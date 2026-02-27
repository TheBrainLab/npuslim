# NPUSlim Framework

## Run Commands

### Quantization
```bash
python tools/run.py -c configs/compressor/int8_dyn/qwen3_0_6b.yaml
python tools/run.py -c configs/compressor/gptq/qwen3_0_6b.yaml
python tools/run.py -c configs/compressor/quip/qwen3_0_6b-w4.yaml
```

### Evaluation
```bash
# LM-Eval (wikitext, ceval, etc.)
bash tools/eval/run_lmeval.sh outputs/qwen_int8_dyn --tasks wikitext -d 0,1 -t 2 -q

# Stress Test with evalscope (requires running vLLM server)
bash tools/eval/evalscope_perf.sh outputs/qwen_int8_dyn --parallel "1 16 32"

# QuIP-specific perplexity evaluation
python tools/eval_ppl_quip_style.py
```

### Deployment (vLLM)
```bash
bash deploy/run_vllm.sh --model-path outputs/qwen3_int8_dyn -d 4,5 -t 2 -q
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
- `TaskFactory.create(task_key, raw_config, resources)` - Creates ptq, eval, save, speculative tasks
- `CompressorFactory.create(algo_name, ...)` - Creates quantization algorithms
- `SaverFactory.create(format_name, ...)` - Creates HuggingFace savers

**Base Classes**:
- `BaseLLMModel`: Model wrapper with HF/ModelScope hub support via `get_hub_class()`
- `BaseTask`: Abstract task with `ConfigClass` auto-parsing and `_resolve_layer_names()` for glob/regex patterns
- `BaseCompressorAlgo`: Unified interface with `prepare()`, `calibrate()`, `convert()`, `compress()`, `apply_masks()`
- `BaseObserver`: Activation/weight observers (AbsMaxActivation, AbsMaxWeight, PTQObserver)

### Pipeline Task Types

- `ptq`: Post-training quantization
- `eval`: Model evaluation (perplexity, accuracy)
- `save`: Export quantized model
- `speculative`: Speculative decoding tasks

### Directory Structure

```
src/npuslim/
├── model/           # BaseLLMModel subclasses (qwen3/, opt/)
├── dataset/         # Calibration datasets (c4_dataset.py, wikitext2_dataset.py, mmlu_dataset.py)
├── tasks/           # BaseTask subclasses (compressor_task.py, eval_task.py, save_task.py)
├── compressor/
│   ├── quantizer/    # BaseCompressorAlgo subclasses (int8_dyn/, gptq/, quip/, sparsegpt/)
│   ├── observers/    # BaseObserver subclasses
│   └── core/        # Quantization utilities, LayerWiseScheduler, quant_func
├── saver/           # Model savers (huggingface.py)
├── vllm_plugin/     # vLLM-ascend integration, updates ASCEND_QUANTIZATION_METHOD_MAP
└── utils/           # config_parser, backend (bh for device-agnostic cache), factory
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
```

## Quantization Algorithms

- **INT8Dynamic** (`src/npuslim/compressor/quantizer/int8_dyn/`): Per-channel weight, per-token activation
- **GPTQ** (`src/npuslim/compressor/quantizer/gptq/`): Activation-aware weight quantization
- **QuIP** (`src/npuslim/compressor/quantizer/quip/`): Quaternion-inspired with vector balancing
- **SparseGPT** (`src/npuslim/compressor/quantizer/sparsegpt/`): Structured pruning + quantization

## vLLM Integration

Package registers via `pyproject.toml` entry point:
```toml
[project.entry-points."vllm.general_plugins"]
register_npuslim = "npuslim.vllm_plugin:register"
```

`vllm_plugin/__init__.py`: Updates `ASCEND_QUANTIZATION_METHOD_MAP` with NPUSLIM quantization methods.

## Critical Notes

- **CANN requirement**: Must set `ASCEND_HOME_PATH` env var before installation
- **Python version**: >=3.11 required
- **Factory lazy loading**: Use `_REGISTRY_MAP` in submodule `__init__.py` for lazy imports
- **Layer patterns**: Tasks support `ignore_layers` with glob (e.g., `output.*`) and regex (prefix `re:`)
- **Backend**: Use `npuslim.utils.backend.bh` for device-agnostic cache clearing (`empty_cache()`)
