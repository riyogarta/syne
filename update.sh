#!/bin/bash
# Syne Update Script — pull latest code and reinstall
set -e

SYNE_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SYNE_DIR"

echo "📥 Pulling latest code..."
git pull

echo "🔧 Setting up virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Installing..."
pip install -e . -q

# Ensure syne is callable from anywhere
SYNE_BIN="$SYNE_DIR/.venv/bin/syne"
if [ -f "$SYNE_BIN" ]; then
    mkdir -p "$HOME/.local/bin"
    ln -sf "$SYNE_BIN" "$HOME/.local/bin/syne"

    # Ensure ~/.local/bin is in PATH
    if ! echo "$PATH" | grep -q "$HOME/.local/bin"; then
        SHELL_RC="$HOME/.bashrc"
        [ -f "$HOME/.zshrc" ] && SHELL_RC="$HOME/.zshrc"
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_RC" 2>/dev/null; then
            echo '' >> "$SHELL_RC"
            echo '# Syne CLI' >> "$SHELL_RC"
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
        fi
        export PATH="$HOME/.local/bin:$PATH"
    fi
fi

echo "✅ Syne updated! Run: syne cli"
