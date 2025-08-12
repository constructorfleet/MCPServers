import argparse
import textwrap
from typing import Dict, Literal, Union, cast
from base import run_server, mcp
from kagiapi import KagiClient
from kagiapi.models import EnrichResponse
from concurrent.futures import ThreadPoolExecutor
import os
import json
import logging
from pydantic import Field

logging.basicConfig(
    level=logging.INFO,  # Use DEBUG for more verbosity during development
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
# These two lines enable debugging at httplib level (requests->urllib3->http.client)
# You will see the REQUEST, including HEADERS and DATA, and RESPONSE with HEADERS but without DATA.
# The only thing missing will be the response.body which is not logged.
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1

# You must initialize logging, otherwise you'll not see debug output.
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
requests_log = logging.getLogger("requests.packages.urllib3")
requests_log.setLevel(logging.DEBUG)
requests_log.propagate = True

kagi_client: KagiClient


@mcp.tool(
    name="get_news",
    description="Retrive new results based on one or more queries using the Kagi Search API.",
    tags={"search", "web", "news"},
)
def search_web(
    query: str = Field(
        description="Concise, keyword-focused search queries. Include essential context within each query for standalone use."
    ),
) -> str:
    """Fetch web results based on one or more queries using the Kagi Search API. Use for general search and when the user explicitly tells you to 'fetch' results/information. Results are from all queries given. They are numbered continuously, so that a user may be able to refer to a result by a specific number."""
    try:
        if not query:
            raise ValueError("Search called with no queries.")
        logger.info(
            "Performing Kagi search for queries: %s %s",
            query,
            repr(kagi_client.session.headers),
        )
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(kagi_client.enrich, [query], timeout=10))
        if not results or any("error" in result for result in results) or len(results) != 1:
            logger.error("Search failed or returned no results: %s", results)
            raise ValueError("Search failed or returned no results")
        results = format_enrich_results(query, results[0])
        logger.info("Search results: %s", results)
        return results

    except Exception as e:
        logger.exception("Error in kagi_search_fetch: %s", e)
        return f"Error: {str(e) or repr(e)}"

mcp.tool(
    name="web_topics",
    description="Fetch web results from the provided query using the Kagi Search API.",
    tags={"search", "web", "news"},
)
def get_news(
    query: str = Field(
        description="Concise, keyword-focused search queries. Include essential context within each query for standalone use."
    ),
) -> str:
    """Fetch web results based on one or more queries using the Kagi Search API. Use for general search and when the user explicitly tells you to 'fetch' results/information. Results are from all queries given. They are numbered continuously, so that a user may be able to refer to a result by a specific number."""
    try:
        if not query:
            raise ValueError("Search called with no queries.")
        logger.info(
            "Performing Kagi search for queries: %s %s",
            query,
            repr(kagi_client.session.headers),
        )
        def enrich_web(query: str) -> EnrichResponse:
            params: Dict[str, Union[int, str]] = {"q": query}

            response = kagi_client.session.get(KagiClient.BASE_URL + "/enrich/news", params=params)
            response.raise_for_status()
            return response.json()
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(enrich_web, [query], timeout=10))
        if not results or any("error" in result for result in results) or len(results) != 1:
            logger.error("Search failed or returned no results: %s", results)
            raise ValueError("Search failed or returned no results")
        results = format_enrich_results(query, results[0])
        logger.info("Search results: %s", results)
        return results

    except Exception as e:
        logger.exception("Error in kagi_search_fetch: %s", e)
        return f"Error: {str(e) or repr(e)}"


def format_search_results(queries: list[str], responses) -> str:
    """Formatting of results for response. Need to consider both LLM and human parsing."""

    result_template = textwrap.dedent("""
        {result_number}: {title}
        {url}
        Published Date: {published}
        {snippet}
    """).strip()

    query_response_template = textwrap.dedent("""
        -----
        Results for search query \"{query}\":
        -----
        {formatted_search_results}
    """).strip()

    per_query_response_strs = []

    start_index = 1
    for query, response in zip(queries, responses):
        # t == 0 is search result, t == 1 is related searches
        results = [result for result in response["data"] if result["t"] == 0]

        # published date is not always present
        formatted_results_list = [
            result_template.format(
                result_number=result_number,
                title=result["title"],
                url=result["url"],
                published=result.get("published", "Not Available"),
                snippet=result["snippet"],
            )
            for result_number, result in enumerate(results, start=start_index)
        ]

        start_index += len(results)

        formatted_results_str = "\n\n".join(formatted_results_list)
        query_response_str = query_response_template.format(
            query=query, formatted_search_results=formatted_results_str
        )
        per_query_response_strs.append(query_response_str)

    return "\n\n".join(per_query_response_strs)

def format_enrich_results(query: str, response: dict) -> str:
    """Formatting of results for response. Need to consider both LLM and human parsing."""

    result_template = textwrap.dedent("""
        {result_number}: {title}
        {url}
        Published Date: {published}
        {snippet}
    """).strip()

    query_response_template = textwrap.dedent("""
        -----
        Results for search query \"{query}\":
        -----
        {formatted_search_results}
    """).strip()

    per_query_response_strs = []

    start_index = 1
    # t == 0 is search result, t == 1 is related searches
    results = [result for result in response["data"] if result["t"] == 0]

    # published date is not always present
    formatted_results_list = [
        result_template.format(
            result_number=result_number,
            title=result["title"],
            url=result["url"],
            published=result.get("published", "Not Available"),
            snippet=result["snippet"],
        )
        for result_number, result in enumerate(results, start=start_index)
    ]

    start_index += len(results)

    formatted_results_str = "\n\n".join(formatted_results_list)
    query_response_str = query_response_template.format(
        query=query, formatted_search_results=formatted_results_str
    )
    per_query_response_strs.append(query_response_str)

    return "\n\n".join(per_query_response_strs)


@mcp.tool()
def kagi_summarizer(
    url: str = Field(description="A URL to a document to summarize."),
    summary_type: Literal["summary", "takeaway"] = Field(
        default="summary",
        description="Type of summary to produce. Options are 'summary' for paragraph prose and 'takeaway' for a bulleted list of key points.",
    ),
    target_language: str | None = Field(
        default=None,
        description="Desired output language using language codes (e.g., 'EN' for English). If not specified, the document's original language influences the output.",
    ),
) -> str:
    """Summarize content from a URL using the Kagi Summarizer API. The Summarizer can summarize any document type (text webpage, video, audio, etc.)"""
    try:
        if not url:
            raise ValueError("Summarizer called with no URL.")

        engine = os.environ.get("KAGI_SUMMARIZER_ENGINE", "cecil")

        valid_engines = {"cecil", "agnes", "daphne", "muriel"}
        if engine not in valid_engines:
            raise ValueError(
                f"Summarizer configured incorrectly, invalid summarization engine set: {engine}. Must be one of the following: {valid_engines}"
            )

        engine = cast(Literal["cecil", "agnes", "daphne", "muriel"], engine)

        summary = kagi_client.summarize(
            url,
            engine=engine,
            summary_type=summary_type,
            target_language=target_language,
        )["data"]["output"]

        logger.info("Summarizer results: %s", summary)

        return summary

    except Exception as e:
        return f"Error: {str(e) or repr(e)}"


def main():
    run_server("Kagi", add_args_fn=add_kagi_args, run_callback=ensure_env)
    
def ensure_env(args):
    if not os.environ.get('KAGI_API_KEY', None):
        if not args.kagi_key:
            raise Exception('Kagi API Key must be provided via --kagi-key or KAGI_API_KEY env')
        os.environ['KAGI_API_KEY'] = args.kagi_key
    global kagi_client
    kagi_client = KagiClient(os.environ["KAGI_API_KEY"])

    return args

def add_kagi_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "-k", "--kagi-api-key",
        type=str,
        default=None,
        help="Authentication token for accessing the Kagi API",
    )
    return parser
    

if __name__ == "__main__":
    main()
