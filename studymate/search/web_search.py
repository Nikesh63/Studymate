"""
Live web search via Tavily, used alongside local documents.
Get a free API key at https://tavily.com (1,000 searches/month free).
"""
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("TAVILY_API_KEY")
MAX_RESULTS = 3  # reduced from 4 -> speed vs thoroughness tradeoff

_client = None


def get_client():
    global _client
    if not API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is not set. Get a free key at https://tavily.com "
            "and add it to a .env file in the project root (see .env.example)."
        )
    if _client is None:
        _client = TavilyClient(api_key=API_KEY)
    return _client


def search_web(query: str, max_results: int = MAX_RESULTS):
    """Returns (context_string, sources_list). sources_list: [{"title":..., "url":...}]"""
    client = get_client()
    response = client.search(query=query, max_results=max_results, search_depth="basic")

    results = response.get("results", [])
    context_parts = []
    sources = []
    for i, r in enumerate(results):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")
        context_parts.append(f"[Source {i+1}: {title}]\n{content}")
        sources.append({"title": title, "url": url})

    context = "\n\n".join(context_parts)
    return context, sources
