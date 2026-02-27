# NPUSlim Documentation

This directory contains the Sphinx documentation for NPUSlim.

## Quick Start

```bash
# Build both Chinese and English versions
make clean && make all-html
```

Output: `build/html/zh_CN/` and `build/html/en/`

## Build Commands

| Command | Description |
|---------|-------------|
| `make clean` | Clean build directory |
| `make gettext` | Generate .pot template files |
| `make html-zh` | Build Chinese version only |
| `make html-en` | Build English version only |
| `make all-html` | Build both zh_CN and en versions |

## Full Build Workflow

When source files are modified, run the complete workflow:

```bash
# Step 1: Generate .pot templates from source
make gettext

# Step 2: Sync .pot to .po files (update/add entries)
bash sync_i18n.sh

# Step 3: Translate .po files (manual step)
# Edit files in locales/zh_CN/LC_MESSAGES/
# Fill in msgstr "" with Chinese translations

# Step 4: Rebuild HTML
make clean && make all-html
```

## Translation Guide

### File Structure

```
locales/
└── zh_CN/
    └── LC_MESSAGES/
        ├── about.po
        ├── index.po
        ├── benchmark/
        │   ├── index.po
        │   └── ...
        ├── reference/
        │   └── ...
        └── tutorials/
            └── ...
```

### Translation Format

Edit `.po` files and fill in `msgstr`:

```po
msgid "About"
msgstr "关于"

msgid "Overview"
msgstr "概述"

msgid "Features"
msgstr "特性"
```

### Check Translation Progress

```bash
# Count untranslated entries
find locales/zh_CN/LC_MESSAGES -name "*.po" -exec sh -c 'echo "=== {} ==="; grep -c "^msgstr \"\"" "$1"' _ {} \;
```

## Prerequisites

```bash
pip install sphinx sphinx-intl myst-parser furo
```

## Directory Structure

```
docs/
├── Makefile           # Build commands
├── sync_i18n.sh       # Translation sync script
├── source/            # Source .md files
│   ├── conf.py        # Sphinx config
│   ├── index.md
│   ├── about.md
│   ├── benchmark/
│   ├── reference/
│   ├── tutorials/
│   └── faq/
├── locales/           # Translation files
│   └── zh_CN/
└── build/             # Output directory
    └── html/
        ├── zh_CN/
        └── en/
```
