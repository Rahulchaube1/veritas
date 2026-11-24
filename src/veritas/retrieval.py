import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .schema import Document, Hit


def tokens(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text.casefold())


def features(query: str, doc: Document) -> list[float]:
    q = set(tokens(query))
    title = set(tokens(doc.title))
    body = set(tokens(doc.text))
    return [
        len(q & title) / max(len(q), 1),
        len(q & body) / max(len(q), 1),
        len(q & body) / max(len(q | body), 1),
    ]


class Index:
    def __init__(self, docs: list[Document], dense=None, rerank=None, weights=None):
        self.docs = docs
        self.dense, self.rerank, self.weights = dense, rerank, weights
        self.vec = self.svd = self.ann = None
        self.post: dict[str, list] = defaultdict(list)
        self.sizes = []
        texts = [d.title + " " + d.text for d in docs]
        for i, text in enumerate(texts):
            words = tokens(text)
            self.sizes.append(len(words))
            for term, count in Counter(words).items():
                self.post[term].append((i, count))
        self.avg = max(1, np.mean(self.sizes)) if docs else 1
        self._cached_search = lru_cache(maxsize=128)(self._search)
        if not docs:
            return
        if dense:
            import faiss

            vectors = dense.run(texts, pooled=True)
            self.ann = faiss.IndexHNSWFlat(vectors.shape[1], 24, faiss.METRIC_INNER_PRODUCT)
            self.ann.hnsw.efSearch = 64
            self.ann.add(vectors)
        else:
            self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=25000)
            if any(x.strip() for x in texts):
                matrix = self.vec.fit_transform(texts)
                dims = min(64, matrix.shape[0] - 1, matrix.shape[1] - 1)
                if dims >= 2:
                    self.svd = TruncatedSVD(n_components=dims, random_state=17, n_iter=5)
                    self.vectors = normalize(self.svd.fit_transform(matrix))
                else:
                    self.vectors = matrix
        # Cache belongs to this snapshot and is discarded when documents are upserted.

    def search(self, query: str, k: int, mode: str = "hybrid") -> tuple[Hit, ...]:
        return self._cached_search(query, k, mode)

    def _search(self, query: str, k: int, mode: str = "hybrid") -> tuple[Hit, ...]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError("Unknown retrieval mode")
        if not self.docs:
            return ()
        n = len(self.docs)
        lexical: Counter = Counter()
        for term in sorted(set(tokens(query))):
            posting = self.post.get(term, [])
            idf = math.log1p((n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, f in posting:
                lexical[i] += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * self.sizes[i] / self.avg))
        depth = min(n, max(32, k))
        ranks = []
        if mode != "dense":
            ranks.append(
                [i for i, _ in sorted(lexical.items(), key=lambda pair: (-pair[1], pair[0]))[:depth]]
            )
        if mode != "bm25":
            if self.ann:
                vals, ids = self.ann.search(self.dense.run([query], pooled=True), depth)
                order = [int(i) for i, v in zip(ids[0], vals[0]) if i >= 0 and v > 0]
            else:
                assert self.vec is not None
                q = self.vec.transform([query])
                if self.svd:
                    q = normalize(self.svd.transform(q))
                sim = self.vectors @ q.T
                scores = np.asarray(sim.toarray() if hasattr(sim, "toarray") else sim).ravel()
                order = [int(i) for i in np.argsort(-scores, kind="stable")[:depth] if scores[i] > 1e-8]
            ranks.append(order)
        fused: dict[int, float] = defaultdict(float)
        for order in ranks:
            for pos, i in enumerate(order, 1):
                fused[i] += 1 / (60 + pos)
        candidates = [i for i, _ in sorted(fused.items(), key=lambda pair: -pair[1])[:depth]]
        cross = {}
        if self.rerank and mode == "hybrid" and candidates:
            logits = self.rerank.run(
                [(query, self.docs[i].title + " " + self.docs[i].text) for i in candidates]
            )
            cross = dict(zip(candidates, logits.ravel().tolist()))
        hits = []
        for i in candidates:
            doc = self.docs[i]
            vals = features(query, doc)
            score = fused[i]
            parts = dict(
                rrf=score, bm25=lexical[i], title_coverage=vals[0], body_coverage=vals[1], jaccard=vals[2]
            )
            if mode == "hybrid" and self.weights:
                logit = float(np.dot(vals, self.weights["coef"]) + self.weights["intercept"])
                parts["learned_relevance"] = float(1 / (1 + np.exp(-np.clip(logit, -40, 40))))
                score += parts["learned_relevance"] / 60
            if cross:
                parts["cross_encoder_logit"] = cross[i]
                score += 0.05 / (1 + math.exp(-max(-40, min(40, cross[i]))))
            hits.append(Hit(doc_id=doc.id, score=score, features=parts))
        return tuple(sorted(hits, key=lambda h: -h.score)[:k])
