# Adding Custom Model Support

This guide shows how to add support for new model architectures to NPUSlim.

## Overview

NPUSlim uses a factory pattern for model creation. To add a new model:

1. Create a model adapter class inheriting from `BaseLLMModel`
2. Register the model in the factory registry
3. Update config parser if needed

## Step 1: Create Model Adapter

Create a new file in `src/npuslim/model/<model_name>/`:

```python
# src/npuslim/model/llama/llama_model.py
from npuslim.model.base_model import BaseLLMModel
from npuslim.utils.config_parser import ModelConfig

class LlamaModel(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config=config)
        self.model_type = "llama"

        # Llama-specific settings
        self.skip_layer_names = ["lm_head", "embed_tokens"]
        self.observer_layer_classes = [torch.nn.Linear]
        self.pre_transformer_module_names = ["model.embed_tokens"]
```

## Step 2: Register Model

Update `src/npuslim/model/llama/__init__.py`:

```python
from npuslim.utils.factory import ModelFactory
from .llama_model import LlamaModel

@ModelFactory.register(name="llama")
class LlamaModelFactory:
    @staticmethod
    def create(config):
        return LlamaModel(config=config)
```

Or use the `_REGISTRY_MAP` for lazy loading:

```python
# src/npuslim/model/llama/__init__.py
_REGISTRY_MAP = {
    "llama": ".llama_model:LlamaModel"
}
```

## Step 3: Update Config

Add your model to the config template:

```yaml
model:
  type: Llama        # New model type
  model_path: meta-llama/Llama-2-7b
  model_hub: hf
```

## Optional: Custom Layer Handling

If your model has special layer types, override `get_observer_layers`:

```python
class LlamaModel(BaseLLMModel):
    def get_observer_layers(self, ignore_layers: list = []):
        # Custom logic for layer selection
        all_modules = dict(self.model.named_modules())
        target_layers = {}

        for name, module in all_modules.items():
            # Add custom layer types here
            if isinstance(module, (torch.nn.Linear, CustomLayer)):
                if ignore_layers and any(ignored in name for ignored in ignore_layers):
                    continue
                target_layers[name] = module

        return target_layers
```

## Test Your Model

```bash
# Create test config
cat > configs/test_llama.yaml << EOF
meta:
  type: llm
model:
  type: Llama
  model_path: meta-llama/Llama-2-7b
pipeline:
  - type: save
    save_dir: ./test_output
EOF

# Run test
python tools/run.py -c configs/test_llama.yaml
```

## See Also

- [BaseLLMModel Reference](../reference/models.md)
- [Quantization Algorithms](../reference/quantizers.md)
