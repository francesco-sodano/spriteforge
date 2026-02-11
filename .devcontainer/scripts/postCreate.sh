#!/usr/bin/env bash
set -euo pipefail

# Default to copy mode to avoid hardlink failures when cache and workspace differ.
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_PATH="$WORKSPACE_DIR/.venv"
PYTHON_VERSION="3.12.12"

# Always ensure the required managed Python is installed (survives container rebuilds)
uv python install "$PYTHON_VERSION"

# Recreate venv if it doesn't exist or points to a wrong/broken Python
if [ -d "$VENV_PATH" ]; then
	CURRENT_PY="$("$VENV_PATH/bin/python" --version 2>/dev/null | awk '{print $2}' || true)"
	# Check major.minor matches (3.12.x)
	if [[ "$CURRENT_PY" != 3.12.* ]]; then
		echo "Existing venv has Python $CURRENT_PY instead of $PYTHON_VERSION — recreating..."
		rm -rf "$VENV_PATH"
	fi
fi

if [ ! -d "$VENV_PATH" ]; then
	uv venv --python "$PYTHON_VERSION" "$VENV_PATH"
fi

. "$VENV_PATH/bin/activate"
# Install from pyproject.toml, including the dev dependency group.
uv sync --group dev

# Auto-activate venv for future shells
if ! grep -q "source $VENV_PATH/bin/activate" /root/.bashrc; then
	printf '\nsource %s/bin/activate\n' "$VENV_PATH" >> /root/.bashrc
fi