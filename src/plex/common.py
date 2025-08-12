import argparse
import os
from typing import Callable
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
