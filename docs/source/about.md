# About

## Overview

NPUSlim is an NPU-oriented model compression and quantization framework designed specifically for Huawei Ascend NPU. It provides efficient post-training quantization algorithms that integrate seamlessly with vLLM-ascend for high-throughput inference.

## Features

- Multiple PTQ algorithms: INT8Dynamic, GPTQ, QuIP, SparseGPT
- NPU-optimized for Ascend hardware
- Modular pipeline system
- Flexible configuration via YAML
- vLLM plugin integration

## License

Apache-2.0

## Citation

```bibtex
@software{npuslim,
  title = {NPUSlim: NPU-oriented Model Compression Framework},
  author = {weiyangdaren},
  year = {2026},
  license = {Apache-2.0},
  url = {https://github.com/TheBrainLab/npuslim}
}
```

## Contributing

We welcome contributions! Feel free to open issues or submit pull requests on our GitHub repository.

## Acknowledgments

- vLLM-ascend team for NPU inference framework
- Original QuIP authors for the quantization algorithm
- Hugging Face for model hub support
