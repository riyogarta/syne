#!/usr/bin/env bash
# Syne Node — Lightweight installer for remote machines.
#
# This installs Syne in "node mode" — a remote CLI that connects
# to your Syne server for LLM and memory, while executing tools locally.
#
# Prerequisites: Python 3.11+, git
#
# Usage:
#   git clone -b remote-node https://github.com/riyogarta/syne.git
#   cd syne
#   bash install-node.sh
#
# After remote-node is merged to main, just:
#   git clone https://github.com/riyogarta/syne.git

set -e

echo ""
echo "  Syne Node — Remote CLI Installer"
echo "  ================================="
echo ""

# Check Python
PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ERROR: Python 3.11+ required but not found."
    echo "  Install: sudo apt install python3.11 python3.11-venv"
    exit 1
fi
echo "  Python: $PYTHON ($($PYTHON --version))"

# Create venv
SYNE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SYNE_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Install
echo "  Installing Syne..."
"$VENV_DIR/bin/pip" install -e "$SYNE_DIR" -q

# Symlink syne to ~/.local/bin
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"
VENV_SYNE="$VENV_DIR/bin/syne"
TARGET="$LOCAL_BIN/syne"

if [ -L "$TARGET" ] || [ -e "$TARGET" ]; then
    rm -f "$TARGET"
fi
ln -s "$VENV_SYNE" "$TARGET"
echo "  Symlink: $TARGET -> $VENV_SYNE"

# Ensure ~/.local/bin is in PATH
SHELL_RC="$HOME/.bashrc"
if [ -f "$HOME/.zshrc" ]; then
    SHELL_RC="$HOME/.zshrc"
fi
if ! grep -q '.local/bin' "$SHELL_RC" 2>/dev/null; then
    echo "" >> "$SHELL_RC"
    echo '# Syne CLI' >> "$SHELL_RC"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_RC"
    echo "  Added ~/.local/bin to PATH in $SHELL_RC"
fi

echo ""
echo "  Installation complete!"
echo ""
echo "  Next steps:"
echo "    1. Run: syne node init"
echo "       (pair with your Syne server using a pairing token)"
echo ""
echo "    2. Run: syne node cli"
echo "       (start the remote CLI)"
echo ""
echo "  If 'syne' command not found, run: source $SHELL_RC"
echo ""
