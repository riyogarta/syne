#!/usr/bin/env bash
# rollback.sh — emergency recovery for Syne production.
# TERMINAL-ONLY. The agent is hard-denied from executing this by shell_guard
# (any invocation of a file literally named rollback.sh -> HARD_DENY). That is
# intentional: recovery must be a conscious human act, never agent-driven.
#
# Place at:  /home/syne/syne/rollback.sh   (chmod +x)
# Run as the syne user:  ~/syne/rollback.sh            (previous commit)
#                        ~/syne/rollback.sh <git-ref>  (explicit target)
#
# What it does:
#   1. Snapshots the current HEAD (so you can roll *forward* again).
#   2. Hard-resets the production checkout to the target ref.
#   3. Reinstalls the package into the venv.
#   4. py_compile-gates the tree — aborts the restart if the target is ALSO
#      broken (prevents restarting into a second bad state).
#   5. Restarts the systemd service.
#
# It does NOT touch the database, .env, credentials, or workspace/.

set -euo pipefail

SYNE_DIR="/home/syne/syne"
VENV_PY="${SYNE_DIR}/.venv/bin/python"
TARGET="${1:-HEAD~1}"

cd "$SYNE_DIR"

echo "==> Current HEAD:"
git rev-parse --short HEAD

PREV="$(git rev-parse HEAD)"
echo "==> Rolling back to: ${TARGET}"

# Resolve target to a concrete commit before touching anything.
TARGET_SHA="$(git rev-parse --verify "${TARGET}^{commit}")"
echo "==> Target resolves to: $(git rev-parse --short "$TARGET_SHA")"

read -r -p "Proceed with hard reset ${PREV:0:7} -> ${TARGET_SHA:0:7}? [y/N] " ans
[[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 1; }

echo "==> Resetting working tree..."
git reset --hard "$TARGET_SHA"

echo "==> Reinstalling into venv..."
"${SYNE_DIR}/.venv/bin/pip" install -e "$SYNE_DIR" --quiet

echo "==> Syntax-gating the target tree (py_compile)..."
if ! "$VENV_PY" -m compileall -q "${SYNE_DIR}/syne"; then
    echo "!! Target tree ${TARGET_SHA:0:7} ALSO fails py_compile."
    echo "!! NOT restarting. You rolled back to another broken state."
    echo "!! Pick an older ref:  ~/syne/rollback.sh <older-sha>"
    echo "!! (Forward again:     ~/syne/rollback.sh ${PREV:0:7})"
    exit 2
fi

echo "==> Restarting service..."
systemctl --user restart syne

sleep 3
if systemctl --user is-active --quiet syne; then
    echo "==> OK. Service active on $(git rev-parse --short HEAD)."
    echo "==> To roll forward again: ~/syne/rollback.sh ${PREV:0:7}"
else
    echo "!! Service did NOT come up. Check: journalctl --user -u syne -n 50"
    exit 3
fi
