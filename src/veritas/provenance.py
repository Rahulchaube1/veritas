"""Build a conservative local graph over examined sources; no network access."""

import re
from collections import defaultdict
from urllib.parse import urlsplit, urlunsplit

from .retrieval import tokens
from .schema import Document, Edge


def canonical(url: str) -> str:
    p = urlsplit(url)
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path.rstrip("/"), p.query, ""))


def graph(docs: list[Document]) -> tuple[dict[str, str], list[Edge]]:
    roots = {d.id: d.id for d in docs}
    by_id = {d.id: d for d in docs}

    def root(i):
        while roots[i] != i:
            roots[i] = roots[roots[i]]
            i = roots[i]
        return i

    buckets: dict[tuple, list[str]] = defaultdict(list)
    shingles: dict[str, set] = {}
    edges = []
    for d in sorted(docs, key=lambda x: (x.published is None, x.published, x.id)):
        words = tokens(d.text)
        grams = set(zip(words, words[1:], words[2:]))
        keys = [("text", tuple(words)), ("ref", d.id)]
        if d.url:
            keys.extend([("url", canonical(d.url)), ("ref", d.url)])
        keys.extend(("ref", ref) for ref in d.cites)
        for quote in re.findall(r'["“]([^"”]{30,1200})["”]', d.text):
            if len(tokens(quote)) >= 7:
                keys.append(("quote", tuple(tokens(quote))))
        if d.publisher:
            keys.append(("publisher", d.publisher.casefold()))
        peers: dict[str, set] = defaultdict(set)
        for key in keys:
            for peer in buckets[key]:
                peers[peer].add(key[0])
        for gram in grams:
            for peer in buckets[("gram", gram)]:
                peers[peer].add("gram")
        for other, shared in sorted(peers.items()):
            sim = len(grams & shingles[other]) / max(1, len(grams | shingles[other]))
            source, target = d.id, other
            if "text" in shared or "url" in shared:
                relation, basis, group = "same_content", "identical normalized text or canonical URL", True
            elif "ref" in shared:
                other_doc = by_id[other]
                if other_doc.id in d.cites or other_doc.url and other_doc.url in d.cites:
                    relation, basis, group = "cites", "declared citation", True
                elif d.id in other_doc.cites or d.url and d.url in other_doc.cites:
                    source, target = other, d.id
                    relation, basis, group = "cites", "declared citation", True
                else:
                    relation, basis, group = "shared_reference", "common declared reference", True
            elif "quote" in shared:
                relation, basis, group = (
                    "shared_quote",
                    "identical quoted passage of at least seven tokens",
                    True,
                )
            elif "publisher" in shared:
                relation, basis, group = "same_publisher", "conservative publisher-level vote cap", True
            elif sim >= 0.48 and len(grams) >= 5:
                relation, basis, group = "possible_derivative", f"trigram Jaccard {sim:.2f}", sim >= 0.78
            else:
                continue
            edges.append(Edge(source=source, target=target, relation=relation, basis=basis, grouped=group))
            if group:
                a, b = sorted([root(d.id), root(other)])
                roots[b] = a
        shingles[d.id] = grams
        for key in set(keys):
            buckets[key].append(d.id)
        for gram in grams:
            buckets[("gram", gram)].append(d.id)
    return {i: root(i) for i in roots}, edges
