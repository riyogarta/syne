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
fi

echo "✅ Syne updated! Run: syne cli"
