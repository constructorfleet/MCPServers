#!/bin/bash
set -e  # Crash fast and loud if anything fails
read -r -a args <<< "$MCP_EXTRA_ARGS"

. .venv/bin/activate

exec uv run ${MCP_PACKAGE} \
    -t "$MCP_TRANSPORT" \
    -b "$MCP_HOST" \
    -p "$MCP_PORT" \
    -m "$MCP_MOUNT" \
    -l "$MCP_LOG_LEVEL" \
    "${args[@]}"
