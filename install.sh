#!/usr/bin/env bash
# Syne installer — handles all dependencies automatically.
# Usage: curl -fsSL https://raw.githubusercontent.com/riyogarta/syne/main/install.sh | bash
#   or:  bash install.sh  (from inside the cloned repo)
set -e

SYNE_DIR="${SYNE_DIR:-$HOME/syne}"
REPO_URL="https://github.com/riyogarta/syne.git"

echo "🧠 Syne Installer"
echo ""

# ── 1. Ensure git ──────────────────────────────────────────
if ! command -v git &>/dev/null; then
    echo "Installing git..."
    sudo apt-get update -qq && sudo apt-get install -y -qq git
fi

# ── 2. Clone repo if not already in it ─────────────────────
if [ -f "pyproject.toml" ] && grep -q "syne" pyproject.toml 2>/dev/null; then
    SYNE_DIR="$(pwd)"
    echo "Using current directory: $SYNE_DIR"
else
    if [ -d "$SYNE_DIR" ] && [ -f "$SYNE_DIR/pyproject.toml" ]; then
        echo "Syne already cloned at $SYNE_DIR"
    else
        echo "Cloning Syne..."
        git clone "$REPO_URL" "$SYNE_DIR"
    fi
    cd "$SYNE_DIR"
fi

# ── 3. Ensure Python 3.11+ ────────────────────────────────
PY=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info >= (3,11))" 2>/dev/null || echo "False")
        if [ "$ver" = "True" ]; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "Python 3.11+ not found. Installing..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip
    PY="python3"
fi

echo "Using Python: $PY ($($PY --version 2>&1))"

# ── 4. Ensure python3-venv ────────────────────────────────
if ! $PY -m venv --help &>/dev/null 2>&1; then
    PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "Installing python${PY_VER}-venv..."
    sudo apt-get update -qq && sudo apt-get install -y -qq "python${PY_VER}-venv"
fi

# ── 5. Create venv ────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PY -m venv .venv
fi

# ── 6. Install Syne ──────────────────────────────────────
echo "Installing Syne..."
.venv/bin/pip install -e . -q

echo ""
echo "✓ Syne installed successfully!"
echo ""
echo "Next: run syne init"
echo ""

# Activate venv for the current shell if interactive
if [ -t 0 ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    exec syne init
fi
