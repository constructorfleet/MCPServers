#!/bin/bash
set -e  # Crash fast and loud if anything fails
read -r -a args <<< "$SCRIPT_ARGS"
. .venv/bin/activate

exec uv run "$SCRIPT" \
    -l "$LOG_LEVEL" \
    "${args[@]}"
