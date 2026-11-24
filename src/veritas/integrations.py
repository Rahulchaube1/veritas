"""Explicit operator-initiated SearXNG JSON import. Uses returned snippets only."""

import hashlib
import json
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .schema import Document


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def searxng(endpoint: str, query: str, limit: int = 8) -> list[Document]:
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Set a fixed HTTPS SearXNG endpoint with no credentials or query")
    if not 1 <= limit <= 20:
        raise ValueError("limit must be 1–20")
    request = Request(
        endpoint + "?" + urlencode({"q": query, "format": "json"}), headers={"Accept": "application/json"}
    )
    with build_opener(NoRedirect).open(request, timeout=12) as response:
        payload = response.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("Provider response exceeds limit")
    docs = []
    for item in json.loads(payload)["results"][:limit]:
        if item.get("content"):
            docs.append(
                Document(
                    id=hashlib.sha256(item["url"].encode()).hexdigest(),
                    url=item["url"],
                    publisher=urlsplit(item["url"]).hostname or "",
                    title=item.get("title", ""),
                    text=item["content"],
                )
            )
    return docs
