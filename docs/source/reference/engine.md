# Engine

## SlimEngine

Main orchestrator for NPUSlim pipeline.

```{eval-rst}
.. autoclass:: npuslim.slim_engine.SlimEngine
   :members:
   :undoc-members:
```

### Workflow

1. **Load Config**: Parse YAML config via `GlobalConfig`
2. **Prepare Resources**: Load models, dataloader, tokenizer
3. **Build Pipeline**: Create task instances from config
4. **Execute Tasks**: Run tasks sequentially with memory cleanup

### Resources

Global resources shared across tasks:

| Resource | Type | Description |
|-----------|--------|-------------|
| `main_model` | BaseLLMModel | Target/teacher model |
| `draft_model` | BaseLLMModel | Draft model for speculative decoding |
| `student_model` | BaseLLMModel | Student model for distillation |
| `dataloader` | DataLoader | Calibration/evaluation data |
| `tokenizer` | Tokenizer | Model tokenizer |
| `engine` | SlimEngine | Engine reference |

## Factory Classes

### ModelFactory

Creates model instances.

```{eval-rst}
.. autoclass:: npuslim.utils.factory.ModelFactory
   :members:
   :undoc-members:
```

### DatasetFactory

Creates dataset instances.

```{eval-rst}
.. autoclass:: npuslim.utils.factory.DatasetFactory
   :members:
   :undoc-members:
```

### TaskFactory

Creates task instances.

```{eval-rst}
.. autoclass:: npuslim.utils.factory.TaskFactory
   :members:
   :undoc-members:
```

### CompressorFactory

Creates quantization algorithm instances.

```{eval-rst}
.. autoclass:: npuslim.utils.factory.CompressorFactory
   :members:
   :undoc-members:
```

### SaverFactory

Creates saver instances.

```{eval-rst}
.. autoclass:: npuslim.utils.factory.SaverFactory
   :members:
   :undoc-members:
```

## Lazy Loading

Factories support lazy loading via `_REGISTRY_MAP`:

```python
# src/npuslim/model/llama/__init__.py
_REGISTRY_MAP = {
    "llama": ".llama_model:LlamaModel"
}
```

This defers import until the model is actually used.

## Usage

```python
from npuslim import SlimEngine

# Run pipeline (loads config from args)
engine = SlimEngine()
engine.run()
```
