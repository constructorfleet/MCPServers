#Dockerfile.base
FROM python:3.11-slim

# Set env defaults; override at runtime or build time
ENV MCP_PACKAGE=
ENV MCP_EXTRA_ARGS=""
ENV MCP_LOG_LEVEL=INFO
ENV MCP_TRANSPORT=stdio
ENV MCP_PORT=3000
ENV MCP_HOST=0.0.0.0
ENV MCP_MOUNT=sse

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

WORKDIR /app

# Copy base package first to leverage layer caching
COPY ./base ./base
COPY ./src ./src

# Build base package into virtualenv
RUN uv venv .venv && \
    . /app/.venv/bin/activate && \
    cd base && \
    uv sync --active

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/app/base/entrypoint.sh"]