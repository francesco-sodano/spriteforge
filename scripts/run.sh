#!/usr/bin/env bash
# Local development entrypoint — runs spriteforge via uv.
set -euo pipefail
exec uv run python -m spriteforge "$@"
