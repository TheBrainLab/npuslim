# NPUSlim v2 Documentation

MkDocs Material documentation for NPUSlim. Source files are Markdown, built into static HTML.

## Build

```bash
# Install dependencies
pip install -r requirements.txt

# Build static HTML to ../site/
mkdocs build -f mkdocs.yml
```

## Live Preview

```bash
# Start dev server with hot-reload at http://127.0.0.1:8000
mkdocs serve -f mkdocs.yml
```

## Directory Layout

```
docs/
├── mkdocs.yml          # MkDocs configuration
├── requirements.txt    # Python build dependencies
├── index.md            # Homepage
├── getting-started.md  # Quick start guide
├── design/             # Architecture & design docs
├── guide/              # Usage guides (config, calibration, quantization)
├── internals/          # Internal mechanisms
├── deployment/         # Serving & evaluation
├── plugins/            # Plugin ecosystem
└── reference/          # CLI & config reference
```
