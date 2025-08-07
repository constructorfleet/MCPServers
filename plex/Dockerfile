ARG BASE_IMAGE="docker.home.prettybaked.com/constructorfleet/mcp-base"
ARG BASE_VERSION=latest
FROM "${BASE_IMAGE}:${BASE_VERSION}" AS base
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl gnupg lsb-release && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends python3.12 python3.12-venv python3.12-dev && \
    ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --upgrade pip

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /uvx /bin/

COPY --from=base /app /app
RUN uv venv .venv && \
    . /app/.venv/bin/activate && \
    cd base && \
    uv sync --active

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["/app/base/entrypoint.sh"]

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