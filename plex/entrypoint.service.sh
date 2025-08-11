#!/bin/bash
set -e  # Crash fast and loud if anything fails

. .venv/bin/activate

exec uv run "$SERVICE_PACKAGE" \
    -l "$LOG_LEVEL" \
    "${args[@]}"
