import asyncio
from contextlib import suppress
from contextlib import suppress
from math import e
import signal
from base import run_daemon
from plex.common import add_args, check_args, check_args
from plex.knowledge import KnowledgeBase
from plex.plex_api import PlexAPI, PlexTextSearch


async def load_data(plex_url: str = "", plex_token: str = "", qdrant_host: str = "", qdrant_port: int = 0, model_name: str = "", sleep: int = 0, log_level: str = "INFO"):
    """Start the Plex service to load data into the knowledge base."""
    plex_search = PlexTextSearch(
        PlexAPI(plex_url, plex_token),
        KnowledgeBase(model_name, qdrant_host, qdrant_port),
    )
    await plex_search.schedule_load_items(sleep)

def main():
    run_daemon("plex-data", load_data, add_args_fn=add_args(plex=True, qdrant=True, sleep=True), run_callback=check_args)

# --- Main Execution ---
if __name__ == "__main__":
    main()