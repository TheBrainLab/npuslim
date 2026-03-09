# file: src/npuslim/compressor/__init__.py


_REGISTRY_MAP = {
    "GPTQ": ".quantizer.gptq.gptq",
    "QuIP": ".quantizer.quip.quip",
    "QuIPSharp": ".quantizer.quip_sharp.quip_sharp",
    "INT8Dynamic": ".quantizer.int8_dyn.int8_dyn",
    "SparseGPT": ".quantizer.sparsegpt.sparsegpt",
}
