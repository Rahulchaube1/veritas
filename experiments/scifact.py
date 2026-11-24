"""Fixed split and ranker. Official dev is touched only by the reporting path."""

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from veritas import Document, Veritas
from veritas.retrieval import features

ROOT = Path(".cache/scifact")


def rows(name):
    return [json.loads(s) for s in (ROOT / (name + ".jsonl")).read_text().splitlines()]


def corpus():
    return [
        Document(id=str(r["doc_id"]), title=r["title"], text="\n".join(r["abstract"])) for r in rows("corpus")
    ]


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def fit(docs, train, dev):
    # Exclude all official-dev cited document IDs and normalized claims before building pairs.
    held = {str(i) for r in dev for i in r["cited_doc_ids"]}
    dev_claims = {r["claim"].lower().strip() for r in dev}
    dev_texts = {d.text for d in docs if d.id in held}
    pool = [d for d in docs if d.id not in held and d.text not in dev_texts]
    by_id = {d.id: d for d in pool}
    rng = random.Random(17)
    data = {"train": ([], []), "validation": ([], []), "calibration": ([], [])}

    def group(i):
        n = int(hashlib.sha256(by_id[i].text.encode()).hexdigest()[:8], 16) % 10
        return "train" if n < 7 else "validation" if n == 7 else "calibration"

    pools = {s: [d for d in pool if group(d.id) == s] for s in data}
    for r in train:
        ids = set(r["evidence"])
        if not ids or r["claim"].lower().strip() in dev_claims or any(i not in by_id for i in ids):
            continue
        splits = {group(i) for i in ids}
        if len(splits) != 1:
            continue
        split = splits.pop()
        X, y = data[split]
        for i in sorted(ids):
            X.append(features(r["claim"], by_id[i]))
            y.append(1)
        for d in rng.sample([d for d in pools[split] if d.id not in ids], 5):
            X.append(features(r["claim"], d))
            y.append(0)
    model = LogisticRegression(max_iter=500, random_state=17).fit(*data["train"])
    checkpoint = {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "features": ["title_coverage", "body_coverage", "jaccard"],
        "seed": 17,
    }
    write("models/ranker.json", checkpoint)
    write(
        "results/ranker_training.json",
        {s: {"pairs": len(y), "log_loss": log_loss(y, model.predict_proba(X))} for s, (X, y) in data.items()},
    )


def metrics(ids, gold, k=10):
    rel = [int(i in gold) for i in ids[:k]]
    positions = [j + 1 for j, v in enumerate(rel) if v]
    dcg = sum(v / math.log2(j + 2) for j, v in enumerate(rel))
    ideal = sum(1 / math.log2(j + 2) for j in range(min(k, len(gold))))
    return [sum(rel) / len(gold), sum(rel) / k, 1 / positions[0] if positions else 0, dcg / ideal]


def evaluate(docs, dev):
    engine = Veritas()
    engine.add(docs)
    results = {}
    evidence = [r for r in dev if r["evidence"]]
    for mode in ["bm25", "dense", "hybrid", "learned_hybrid"]:
        if mode == "learned_hybrid":
            engine.weights = json.loads(Path("models/ranker.json").read_text())
            engine.index = None
        values = []
        lat = []
        for r in evidence:
            t = time.perf_counter()
            hits = engine.retrieve(r["claim"], 10, "hybrid" if mode == "learned_hybrid" else mode)
            lat.append(1000 * (time.perf_counter() - t))
            values.append(metrics([h.doc_id for h in hits], set(r["evidence"])))
        arr = np.array(values)
        rng = np.random.default_rng(17)
        ci = np.quantile(arr[rng.integers(len(arr), size=(1500, len(arr)))].mean(1), [0.025, 0.975], axis=0)
        results[mode] = {
            name: {"mean": float(m), "ci95": v.tolist()}
            for name, m, v in zip(["recall@10", "precision@10", "mrr@10", "ndcg@10"], arr.mean(0), ci.T)
        }
        results[mode]["query_ms_including_first_index"] = dict(
            zip(["p50", "p95", "p99"], np.percentile(lat, [50, 95, 99]).tolist())
        )
    # End-to-end rule verifier, including insufficient-evidence claims.
    verdicts = []
    for r in dev:
        res = engine.search(r["claim"], max_search_steps=1)
        truth = {a["label"] for v in r["evidence"].values() for a in v}
        gold = (
            "insufficient"
            if not truth
            else "conflict"
            if len(truth) > 1
            else {"SUPPORT": "supported", "CONTRADICT": "refuted"}[truth.pop()]
        )
        verdicts.append((res.claims[0].verdict, gold))
    results["verification"] = {
        "n": len(verdicts),
        "answered": sum(a in {"supported", "refuted"} for a, b in verdicts),
        "correct": sum(a == b for a, b in verdicts),
        "always_insufficient_correct": sum(b == "insufficient" for a, b in verdicts),
    }
    write(
        "results/scifact.json",
        {
            "documents": len(docs),
            "retrieval_queries": len(evidence),
            "split": "official dev with gold rationales",
            "results": results,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fit", "evaluate", "all"], default="all", nargs="?")
    args = p.parse_args()
    docs = corpus()
    train, dev = rows("claims_train"), rows("claims_dev")
    if args.stage != "evaluate":
        fit(docs, train, dev)
    if args.stage != "fit":
        evaluate(docs, dev)
