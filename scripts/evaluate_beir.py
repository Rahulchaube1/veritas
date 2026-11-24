"""Evaluate a locally obtained official BEIR-format collection under its own license."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from veritas import Config, Document, Veritas


def evaluate(root: Path, split: str, k: int):
    eng = Veritas(Config(corpus_limit=50000, corpus_byte_limit=1_000_000_000))

    def read(name):
        with (root / name).open() as f:
            yield from (json.loads(line) for line in f)

    eng.add(
        Document(id=str(r["_id"]), title=r.get("title", ""), text=r["text"]) for r in read("corpus.jsonl")
    )
    qrels = defaultdict(dict)
    with (root / "qrels" / (split + ".tsv")).open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    out = []
    for q in read("queries.jsonl"):
        rel = qrels.get(str(q["_id"]))
        if not rel:
            continue
        ids = [h.doc_id for h in eng.retrieve(q["text"], k)]
        gains = [rel.get(i, 0) for i in ids]
        ranked = np.array(gains)

        def dcg(g):
            return sum((2**n - 1) / np.log2(j + 2) for j, n in enumerate(g))

        positions = np.flatnonzero(ranked > 0)
        out.append(
            [
                len(positions) / len(rel),
                len(positions) / k,
                1 / (positions[0] + 1) if len(positions) else 0,
                dcg(gains) / dcg(sorted(rel.values(), reverse=True)[:k]),
            ]
        )
    eng.close()
    if not out:
        raise ValueError("No queries with positive relevance judgments")
    return dict(
        queries=len(out), k=k, metrics=dict(zip(["recall", "precision", "mrr", "ndcg"], np.mean(out, axis=0)))
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("directory", type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    print(json.dumps(evaluate(args.directory, args.split, args.k), indent=2))
