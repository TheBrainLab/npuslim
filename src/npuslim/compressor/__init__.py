# file: src/npuslim/compressor/__init__.py


_REGISTRY_MAP = {
    "GPTQ": ".quantizer.gptq.gptq",
    "INT8Dynamic": ".quantizer.int8_dyn.int8_dyn",
    "SparseGPT": ".quantizer.sparsegpt.sparsegpt",
}
