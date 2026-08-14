"""
Local Search Server for Search-R1

This module provides a local search engine interface that mimics the google_search_server.py API.
It sends requests to a local retrieval server (e.g., running retrieval_server.py from Search-R1)
and formats the results to match the expected output format.

Usage:
    In your generate_with_search.py, replace:
        from google_search_server import google_search
    with:
        from local_search_server import local_search as google_search

    And update SEARCH_R1_CONFIGS:
        SEARCH_R1_CONFIGS = {
            "search_url": "http://127.0.0.1:8000/retrieve",  # URL of local retrieval server
            "topk": 3,
            ...
        }
"""

import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


class LocalSearchError(RuntimeError):
    """The retriever request failed or returned a malformed payload."""


def _extract_contexts(payload: object) -> list[dict]:
    """Normalize the retriever's one-query response to Search-R1 documents."""
    if not isinstance(payload, dict):
        raise LocalSearchError(f"retriever response must be an object, got {type(payload).__name__}")

    result_groups = payload.get("result")
    if not isinstance(result_groups, list) or len(result_groups) != 1:
        raise LocalSearchError("retriever response must contain one result group for the one submitted query")

    retrieval_results = result_groups[0]
    if not isinstance(retrieval_results, list):
        raise LocalSearchError("retriever result group must be a list")

    contexts = []
    for index, item in enumerate(retrieval_results):
        if not isinstance(item, dict):
            raise LocalSearchError(f"retriever result {index} must be an object")

        # The Miles retriever returns {"document": {"contents": ...}}.  Some
        # upstream Search-R1 servers return the same contents field flattened at
        # the hit level, so accept both without fabricating placeholder passages.
        document = item.get("document")
        content = document.get("contents") if isinstance(document, dict) else item.get("contents")
        if not isinstance(content, str) or not content.strip():
            raise LocalSearchError(f"retriever result {index} has no non-empty document contents")
        contexts.append({"document": {"contents": content}})

    return contexts


async def local_search(
    search_url: str,
    query: str,
    top_k: int = 5,
    timeout: int = 60,
    proxy: str | None = None,
    max_attempts: int = 3,
) -> list[dict]:
    """
    Call local search engine server and format results to match google_search_server.py output.

    This function provides the same interface as google_search() from google_search_server.py,
    making it a drop-in replacement. The only difference is that instead of using an API key,
    it uses a search_url parameter.

    Args:
        search_url: URL of the local retrieval server (e.g., "http://127.0.0.1:8000/retrieve")
        query: Search query string
        top_k: Number of results to retrieve
        timeout: Request timeout in seconds (default: 60)
        proxy: Proxy URL if needed (normally unused for a local retriever)
        max_attempts: Total attempts for transient HTTP or JSON failures

    Returns:
        List of dictionaries with format: [{"document": {"contents": '"<title>"\n<text>'}}]
        This matches the output format of google_search() from google_search_server.py
    """
    # Prepare request payload for local retrieval server
    payload = {
        "queries": [query],
        "topk": top_k,
        "return_scores": False,  # We don't need scores for compatibility with google_search_server
    }

    if max_attempts < 1:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    # Send async request to local retrieval server. A transient failure aborts
    # the trajectory after these bounded retries; it must never turn into an
    # empty observation because that silently trains without retrieval.
    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    search_url,
                    json=payload,
                    timeout=timeout_obj,
                    proxy=proxy,
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
            return _extract_contexts(result)
        except LocalSearchError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            delay = min(2 ** (attempt - 1), 5)
            logger.warning(
                "Search-R1 retriever request failed (attempt %d/%d): %s; retrying in %ds",
                attempt,
                max_attempts,
                error,
                delay,
            )
            await asyncio.sleep(delay)

    raise LocalSearchError(
        f"retriever request to {search_url} failed after {max_attempts} attempts: {last_error}"
    ) from last_error
