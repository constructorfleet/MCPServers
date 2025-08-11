import asyncio
from contextlib import suppress
from contextlib import suppress
from math import e
import logging
import signal
import click

from plex.knowledge import KnowledgeBase
from plex.plex_api import PlexAPI, PlexTextSearch


async def start_service(plex_url: str = "", plex_token: str = "", qdrant_host: str = "", qdrant_port: int = 0, model_name: str = "", sleep: int = 0, log_level: str = "INFO"):
    """Start the Plex service to load data into the knowledge base."""
    plex_api = PlexAPI(plex_url, plex_token)
    plex_search = PlexTextSearch(
        plex_api,
        KnowledgeBase(model_name, qdrant_host, qdrant_port),
    )
    task = asyncio.create_task(plex_search.schedule_load_items(sleep))
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    await stop.wait()
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
        await plex_api.close()
        await plex_search.close()

@click.command("load-data")
@click.option("-l", "--log-level", type=str, help="Log level", envvar="LOG_LEVEL", default="INFO")
@click.option("-u", "--plex-url", type=str, help="Base URL for the Plex server", envvar="PLEX_URL")
@click.option("-k", "--plex-token", type=str, help="Plex server authentication token", envvar="PLEX_TOKEN")
@click.option("-qh", "--qdrant-host", type=str, default="localhost", help="Qdrant host", envvar="QDRANT_HOST")
@click.option("-qp", "--qdrant-port", type=int, default=6333, help="Qdrant port", envvar="QDRANT_PORT")
@click.option(
    "-mn",
    "--model-name",
    type=str,
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Model name",
    envvar="MODEL_NAME"
)
@click.option("-s", "--sleep", type=int, default=60, help="Sleep duration between tasks (minutes)", envvar="SLEEP_DURATION")
def load_data(plex_url: str = "", plex_token: str = "", qdrant_host: str = "", qdrant_port: int = 0, model_name: str = "", sleep: int = 0, log_level: str = "INFO"):
    # You must initialize logging, otherwise you'll not see debug output.
    logging.basicConfig(
        level=log_level,  # Use DEBUG for more verbosity during development
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    requests_log = logging.getLogger("requests.packages.urllib3")
    requests_log.setLevel(logging.INFO)
    asyncio.run(start_service(plex_url, plex_token, qdrant_host, qdrant_port, model_name, sleep, log_level))


def main():
    load_data()


if __name__ == "__main__":
    main()
