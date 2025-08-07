ARG BASE_IMAGE="docker.home.prettybaked.com/constructorfleet/mcp-base"
ARG BASE_VERSION=latest
FROM "${BASE_IMAGE}:${BASE_VERSION}"

ENV MCP_PACKAGE="mcp-kagi"
ENV KAGI_API_KEY=
# ENV MCP_EXTRA_ARGS="--kagi-api-key $$KAGI_API_KEY"
ENV PROJECT_ROOT="/app"

WORKDIR /app
COPY ./kagi ./kagi

RUN . /app/.venv/bin/activate && \
    cd kagi && \
    uv sync --active

# RUN uv pip install -e /app/src/kagi