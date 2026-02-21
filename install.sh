#!/usr/bin/env bash
# Syne installer — handles all dependencies automatically.
# Usage: bash install.sh  (from inside the cloned repo)
set -e

SYNE_DIR="${SYNE_DIR:-$HOME/syne}"
REPO_URL="https://github.com/riyogarta/syne.git"

echo "🧠 Syne Installer"
echo ""

# ── 0. Pre-cache sudo credentials ─────────────────────────
# Some system packages need sudo. Ask for password upfront
# so it doesn't surprise the user mid-install.
if command -v sudo &>/dev/null; then
    echo "Some system packages may need to be installed."
    echo "Please enter your sudo password if prompted:"
    echo ""
    sudo -v || { echo "sudo failed. Run as a user with sudo access."; exit 1; }
    echo ""
    # Keep sudo alive in background
    ( while true; do sudo -n true; sleep 50; done 2>/dev/null ) &
    SUDO_KEEPALIVE_PID=$!
    trap 'kill $SUDO_KEEPALIVE_PID 2>/dev/null' EXIT
fi

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

# ── 4. Ensure python3-venv and pip ────────────────────────
PY_VER=$($PY -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PKGS=""

# Test venv by actually trying to create one (--help lies about ensurepip)
TEST_VENV=$(mktemp -d)
if ! $PY -m venv "$TEST_VENV/test" &>/dev/null 2>&1; then
    PKGS="python${PY_VER}-venv"
fi
rm -rf "$TEST_VENV"

if ! $PY -m pip --version &>/dev/null 2>&1; then
    PKGS="$PKGS python3-pip"
fi

if [ -n "$PKGS" ]; then
    echo "Installing system packages:$PKGS"
    sudo apt-get update -qq && sudo apt-get install -y -qq $PKGS
    echo "✓ System packages installed"
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

# Run syne init if interactive terminal
if [ -t 0 ]; then
    source .venv/bin/activate
    exec syne init
else
    echo "Next: source .venv/bin/activate && syne init"
fi
