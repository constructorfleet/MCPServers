import argparse
import json
import os
from pathlib import Path
from typing import Callable, Iterable

from base import EnvDefault

def add_plex_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-u",
        "--plex-url",
        action=EnvDefault,
        envvar="PLEX_SERVER_URL",
        type=str,
        help="Base URL for the Plex server (e.g., http://localhost:32400)",
    )
    parser.add_argument(
        "-k",
        "--plex-token",
        type=str,
        action=EnvDefault,
        envvar="PLEX_TOKEN",
        help="Authentication token for accessing the Plex server",
    )
    parser.add_argument(
        "--player-alias",
        action="append",
        metavar="ALIAS=TARGET",
        help="Define an alias for a Plex client (can be used multiple times).",
    )
    parser.add_argument(
        "--player-alias-file",
        type=str,
        help="Path to a JSON file containing player alias mappings.",
    )
    return parser

def add_qdrant_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-qh",
        "--qdrant-host",
        type=str,
        action=EnvDefault,
        envvar="QDRANT_HOST",
        default="localhost",
        help="Host for the Qdrant vector database (default: localhost)",
    )
    parser.add_argument(
        "-qp",
        "--qdrant-port",
        type=int,
        action=EnvDefault,
        envvar="QDRANT_PORT",
        default=6333,
        help="Port for the Qdrant vector database (default: 6333)",
    )
    parser.add_argument(
        "-mn",
        "--model-name",
        type=str,
        action=EnvDefault,
        envvar="MODEL_NAME",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name for the Qdrant vector database (default: sentence-transformers/all-MiniLM-L6-v2)",
    )

    return parser

def add_sleep_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-s",
        "--sleep",
        type=int,
        action=EnvDefault,
        envvar="SLEEP_DURATION",
        default=60,
        help="Sleep duration between tasks in minutes (default: 60)",
    )
    return parser

def _normalize_alias_targets(raw: object) -> list[str]:
    if isinstance(raw, str):
        value = raw.strip()
        return [value] if value else []
    if isinstance(raw, Iterable):
        values: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                values.append(text)
        return values
    raise TypeError("Alias values must be a string or iterable of strings")


def _merge_alias_data(target: dict[str, list[str]], updates: dict[str, object]) -> None:
    for alias, raw_values in updates.items():
        alias_key = alias.strip()
        if not alias_key:
            continue
        try:
            values = _normalize_alias_targets(raw_values)
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError(f"Invalid alias mapping for '{alias_key}'") from exc
        if not values:
            continue
        existing = target.setdefault(alias_key, [])
        for value in values:
            if value not in existing:
                existing.append(value)


def _load_alias_file(file_path: str) -> dict[str, list[str]]:
    data = json.loads(Path(file_path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Alias file must contain a JSON object of alias mappings")
    alias_data: dict[str, list[str]] = {}
    for alias, values in data.items():
        _merge_alias_data(alias_data, {alias: values})
    return alias_data


def _parse_alias_pairs(pairs: Iterable[str]) -> dict[str, list[str]]:
    alias_map: dict[str, list[str]] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError("Player aliases must be provided as ALIAS=TARGET")
        alias, target = pair.split("=", 1)
        alias_key = alias.strip()
        target_value = target.strip()
        if not alias_key or not target_value:
            raise ValueError("Player aliases must include both alias and target name")
        alias_map.setdefault(alias_key, [])
        if target_value not in alias_map[alias_key]:
            alias_map[alias_key].append(target_value)
    return alias_map


def add_args(plex: bool = True, qdrant: bool = True, sleep: bool = True) -> Callable[[argparse.ArgumentParser], argparse.ArgumentParser]:
    """
    Returns a function that adds Plex and Qdrant arguments to the parser.
    """

    def wrapper(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if plex:
            parser = add_plex_args(parser)
        if qdrant:
            parser = add_qdrant_args(parser)
        if sleep:
            parser = add_sleep_arg(parser)
        return parser
    return wrapper


def check_args(args):
    if not os.environ.get("PLEX_SERVER_URL"):
        if not args.plex_url:
            raise ValueError(
                "Plex server URL must be provided via --plex-url or PLEX_SERVER_URL environment variable."
            )
        os.environ["PLEX_SERVER_URL"] = args.plex_url
    if not os.environ.get("PLEX_TOKEN"):
        if not args.plex_token:
            raise ValueError(
                "Plex token must be provided via --plex-token or PLEX_TOKEN environment variable."
            )
        os.environ["PLEX_TOKEN"] = args.plex_token
    if not os.environ.get("QDRANT_HOST"):
        if not args.qdrant_host:
            raise ValueError(
                "Qdrant host must be provided via --qdrant-host or QDRANT_HOST environment variable."
            )
        os.environ["QDRANT_HOST"] = args.qdrant_host
    if not os.environ.get("QDRANT_PORT"):
        if not args.qdrant_port:
            raise ValueError(
                "Qdrant port must be provided via --qdrant-port or QDRANT_PORT environment variable."
            )
        os.environ["QDRANT_PORT"] = str(args.qdrant_port)
    if not os.environ.get("MODEL_NAME"):
        if not args.model_name:
            raise ValueError(
                "Model name must be provided via --model-name or MODEL_NAME environment variable."
            )
        os.environ["MODEL_NAME"] = args.model_name

    alias_map: dict[str, list[str]] = {}
    existing_aliases = os.environ.get("PLEX_PLAYER_ALIASES")
    if existing_aliases:
        try:
            env_aliases = json.loads(existing_aliases)
        except json.JSONDecodeError as exc:
            raise ValueError("PLEX_PLAYER_ALIASES must be valid JSON") from exc
        if not isinstance(env_aliases, dict):
            raise ValueError("PLEX_PLAYER_ALIASES must be a JSON object")
        _merge_alias_data(alias_map, env_aliases)
    if getattr(args, "player_alias_file", None):
        alias_file_map = _load_alias_file(args.player_alias_file)
        _merge_alias_data(alias_map, alias_file_map)
    if getattr(args, "player_alias", None):
        alias_pairs = _parse_alias_pairs(args.player_alias)
        _merge_alias_data(alias_map, alias_pairs)
    if alias_map:
        os.environ["PLEX_PLAYER_ALIASES"] = json.dumps(alias_map)
