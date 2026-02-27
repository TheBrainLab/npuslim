#!/bin/bash

# Usage: ./sync_i18n.sh [POT_DIR] [LOCALE_DIR] [LANG]
# Example: ./sync_i18n.sh docs/build/gettext docs/locales zh_CN

# 1. Parse arguments with defaults
POT_SRC="${1:-docs/build/gettext}"
LOCALE_TARGET="${2:-docs/locales}"
TARGET_LANG="${3:-zh_CN}"

echo "------------------------------------------"
echo "🛠️  I18N Configuration:"
echo "📂 POT Source:   $POT_SRC"
echo "📂 Output Dir:   $LOCALE_TARGET"
echo "🌐 Target Lang:  $TARGET_LANG"
echo "------------------------------------------"

# Check if source directory exists
if [ ! -d "$POT_SRC" ]; then
    echo "❌ Error: Source directory '$POT_SRC' not found. Run 'make gettext' first."
    exit 1
fi

# Ensure output directory exists
mkdir -p "$LOCALE_TARGET"

# 2. Recursively find and process all .pot files
find "$POT_SRC" -name "*.pot" | while read -r pot_file; do
    # Extract relative path, preserving hierarchy (e.g., subdir/file.pot)
    rel_path=$(realpath --relative-to="$POT_SRC" "$pot_file")

    # Preserve directory structure as domain name
    # This allows $domain to contain paths like "usage/install"
    domain="${rel_path%.pot}"

    # Construct .po target path with hierarchy
    # Path: [LOCALE_TARGET]/[TARGET_LANG]/LC_MESSAGES/[SUBDIR]/[FILE].po
    po_file="$LOCALE_TARGET/$TARGET_LANG/LC_MESSAGES/$domain.po"

    echo "📄 Processing: $rel_path"

    # Ensure subdirectory exists
    mkdir -p "$(dirname "$po_file")"

    if [ ! -f "$po_file" ]; then
        echo "   ✨ [INIT] Creating new translation: $domain"
        # Use -o to explicitly specify output path
        pybabel init -i "$pot_file" -o "$po_file" -l "$TARGET_LANG" > /dev/null 2>&1
    else
        echo "   🔄 [UPDATE] Updating existing translation: $domain"
        # Use -o to explicitly specify output path
        pybabel update -i "$pot_file" -o "$po_file" > /dev/null 2>&1
    fi
done

# 3. Compile all translations
echo "------------------------------------------"
echo "🚀 Compiling all translations (.po -> .mo)..."

# Manually iterate and compile to avoid pybabel catalog lookup issues
find "$LOCALE_TARGET/$TARGET_LANG/LC_MESSAGES" -name "*.po" | while read -r po_item; do
    mo_item="${po_item%.po}.mo"
    pybabel compile -i "$po_item" -o "$mo_item" > /dev/null 2>&1
done

echo "✅ Done! Translation files are ready with preserved hierarchy."
