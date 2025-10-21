import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import importlib.util
import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def load_module(name: str, relative_path: str):
    module_path = SRC_DIR / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def install_stubs() -> dict[str, types.ModuleType]:
    modules: dict[str, types.ModuleType] = {}

    class _FastMCP:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

        def add_middleware(self, *args, **kwargs):
            pass

        async def run_async(self, *args, **kwargs):
            pass

        def custom_route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def tool(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    fastmcp_module = types.ModuleType("fastmcp")
    fastmcp_module.FastMCP = _FastMCP
    modules["fastmcp"] = fastmcp_module

    fastmcp_server_module = types.ModuleType("fastmcp.server")
    modules["fastmcp.server"] = fastmcp_server_module

    fastmcp_server_server_module = types.ModuleType("fastmcp.server.server")
    fastmcp_server_server_module.Transport = object
    modules["fastmcp.server.server"] = fastmcp_server_server_module

    fastmcp_server_middleware_module = types.ModuleType("fastmcp.server.middleware")
    modules["fastmcp.server.middleware"] = fastmcp_server_middleware_module

    fastmcp_logging_module = types.ModuleType("fastmcp.server.middleware.logging")

    class _LoggingMiddleware:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    fastmcp_logging_module.LoggingMiddleware = _LoggingMiddleware
    modules["fastmcp.server.middleware.logging"] = fastmcp_logging_module

    modules["mcp"] = types.ModuleType("mcp")
    mcp_types_module = types.ModuleType("mcp.types")

    class _ToolAnnotations:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    mcp_types_module.ToolAnnotations = _ToolAnnotations
    modules["mcp.types"] = mcp_types_module

    plexapi_module = types.ModuleType("plexapi")
    plexapi_module.__path__ = []
    modules["plexapi"] = plexapi_module

    plexapi_base_module = types.ModuleType("plexapi.base")
    plexapi_base_module.PlexSession = type("PlexSession", (), {})
    modules["plexapi.base"] = plexapi_base_module

    plexapi_client_module = types.ModuleType("plexapi.client")
    plexapi_client_module.PlexClient = type("PlexClient", (), {})
    modules["plexapi.client"] = plexapi_client_module

    plexapi_utils_module = types.ModuleType("plexapi.utils")

    def _to_json(value):  # pragma: no cover - simple stub
        return value

    plexapi_utils_module.toJson = _to_json
    modules["plexapi.utils"] = plexapi_utils_module

    plexapi_video_module = types.ModuleType("plexapi.video")
    plexapi_video_module.Video = type("Video", (), {})
    modules["plexapi.video"] = plexapi_video_module

    plexapi_exceptions_module = types.ModuleType("plexapi.exceptions")
    plexapi_exceptions_module.NotFound = Exception
    modules["plexapi.exceptions"] = plexapi_exceptions_module

    pydantic_module = types.ModuleType("pydantic")

    def _field(*args, **kwargs):  # pragma: no cover - simple stub
        return None

    pydantic_module.Field = _field
    modules["pydantic"] = pydantic_module

    rapidfuzz_module = types.ModuleType("rapidfuzz")
    rapidfuzz_process_module = types.ModuleType("rapidfuzz.process")

    def _extract_one(query, choices, score_cutoff=0):  # pragma: no cover - simple stub
        return (choices[0] if choices else None, 100, None)

    rapidfuzz_process_module.extractOne = _extract_one
    modules["rapidfuzz"] = rapidfuzz_module
    modules["rapidfuzz.process"] = rapidfuzz_process_module

    starlette_module = types.ModuleType("starlette")
    modules["starlette"] = starlette_module

    starlette_requests_module = types.ModuleType("starlette.requests")
    starlette_requests_module.Request = type("Request", (), {})
    modules["starlette.requests"] = starlette_requests_module

    starlette_responses_module = types.ModuleType("starlette.responses")

    class _Response:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    class _JSONResponse(_Response):
        pass

    starlette_responses_module.Response = _Response
    starlette_responses_module.JSONResponse = _JSONResponse
    modules["starlette.responses"] = starlette_responses_module

    plex_package = types.ModuleType("plex")
    plex_package.__path__ = [str(SRC_DIR / "plex")]
    modules["plex"] = plex_package

    plex_format_module = types.ModuleType("plex.format")

    def _format(*args, **kwargs):  # pragma: no cover - simple stub
        return ""

    plex_format_module.format_client = _format
    plex_format_module.format_episode = _format
    plex_format_module.format_movie = _format
    plex_format_module.format_session = _format
    modules["plex.format"] = plex_format_module

    plex_knowledge_module = types.ModuleType("plex.knowledge")

    class _KnowledgeBase:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

        def find_media_tool(self, *args, **kwargs):
            pass

    class _PlexMediaQuery:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

    plex_knowledge_module.KnowledgeBase = _KnowledgeBase
    plex_knowledge_module.PlexMediaQuery = _PlexMediaQuery
    modules["plex.knowledge"] = plex_knowledge_module

    plex_plex_api_module = types.ModuleType("plex.plex_api")

    class _PlexAPI:  # pragma: no cover - simple stub
        async def get_sessions(self):
            return []

        async def get_clients(self):
            return []

        async def get_client(self, machine_identifier):
            return None

        async def get_item(self, key):
            return None

        def get_media(self, key):
            return None

    class _PlexTextSearch:  # pragma: no cover - simple stub
        def __init__(self, *args, **kwargs):
            pass

        async def find_media(self, *args, **kwargs):
            return []

    plex_plex_api_module.PlexAPI = _PlexAPI
    plex_plex_api_module.PlexTextSearch = _PlexTextSearch
    modules["plex.plex_api"] = plex_plex_api_module

    for name, module in modules.items():
        sys.modules.setdefault(name, module)

    modules["mcp"].types = mcp_types_module
    fastmcp_module.server = fastmcp_server_module
    fastmcp_server_module.server = fastmcp_server_server_module
    fastmcp_server_module.middleware = fastmcp_server_middleware_module
    fastmcp_server_middleware_module.logging = fastmcp_logging_module
    rapidfuzz_module.process = rapidfuzz_process_module
    starlette_module.requests = starlette_requests_module
    starlette_module.responses = starlette_responses_module
    plexapi_module.base = plexapi_base_module
    plexapi_module.client = plexapi_client_module
    plexapi_module.utils = plexapi_utils_module
    plexapi_module.video = plexapi_video_module
    plexapi_module.exceptions = plexapi_exceptions_module

    load_module("base", "base/__init__.py")

    return modules


STUB_MODULES = install_stubs()

common = load_module("plex.common", "plex/common.py")
check_args = common.check_args

server = load_module("plex.server", "plex/server.py")
set_player_aliases = server.set_player_aliases
match_client_name = server.match_client_name


@pytest.fixture(scope="module", autouse=True)
def cleanup_stubs():  # pragma: no cover - test cleanup
    yield
    for name in STUB_MODULES:
        sys.modules.pop(name, None)
    for extra in ("base", "plex.common", "plex.server"):
        sys.modules.pop(extra, None)


class DummyClient:
    def __init__(self, title: str, machine_identifier: str, capabilities: list[str]):
        self.title = title
        self.machineIdentifier = machine_identifier
        self.protocolCapabilities = capabilities


def make_args(**overrides):
    defaults = dict(
        plex_url="http://localhost:32400",
        plex_token="token",
        qdrant_host="localhost",
        qdrant_port=6333,
        model_name="model",
        player_alias=None,
        player_alias_file=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


@pytest.fixture
def empty_env(monkeypatch):
    for var in [
        "PLEX_SERVER_URL",
        "PLEX_TOKEN",
        "QDRANT_HOST",
        "QDRANT_PORT",
        "MODEL_NAME",
        "PLEX_PLAYER_ALIASES",
    ]:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_check_args_merges_player_alias_sources(tmp_path, empty_env):
    alias_file = tmp_path / "aliases.json"
    alias_file.write_text(json.dumps({"kitchen": "Kitchen TV"}))

    args = make_args(
        player_alias=["lounge=Living Room TV"],
        player_alias_file=str(alias_file),
    )

    check_args(args)

    assert "PLEX_PLAYER_ALIASES" in os.environ
    assert json.loads(os.environ["PLEX_PLAYER_ALIASES"]) == {
        "lounge": ["Living Room TV"],
        "kitchen": ["Kitchen TV"],
    }


def test_match_client_name_uses_alias(empty_env):
    try:
        set_player_aliases({"lounge": ["Living Room TV"]})
        client = DummyClient("Living Room TV", "machine-123", ["playback"])

        result = match_client_name(
            "lounge",
            [client],
            filter=lambda c: "playback" in c.protocolCapabilities,
        )

        assert result is client
    finally:
        set_player_aliases({})
