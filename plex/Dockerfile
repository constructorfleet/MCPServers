ARG BASE_IMAGE="docker.home.prettybaked.com/constructorfleet/mcp-base"
ARG BASE_VERSION=latest
FROM "${BASE_IMAGE}:${BASE_VERSION}"
ENV MCP_PACKAGE="mcp-plex"
ENV PLEX_TOKEN=
ENV PLEX_URL=
# ENV MCP_EXTRA_ARGS="--plex-token $$PLEX_TOKEN --plex-url $$PLEX_URL"
ENV PROJECT_ROOT="/app"

WORKDIR /app
COPY ./plex ./plex

RUN . /app/.venv/bin/activate && \
    cd plex && \
    uv sync --active

# RUN uv --active pip install -e /app/src/plex