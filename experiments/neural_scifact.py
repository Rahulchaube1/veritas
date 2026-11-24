"""End-to-end neural run plus retrieval and passage-grounding diagnostics."""

import random
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scifact import corpus, metrics, rows, write

from veritas import Config, Veritas


def main(confirmatory=False):
    docs = corpus()
    claims = random.Random(17).sample(rows("claims_dev"), 48)
    gate = None
    if confirmatory:
        initial = {r["id"] for r in claims}
        claims = random.Random(41).sample([r for r in rows("claims_dev") if r["id"] not in initial], 80)
        gate = Path("models/rationale_gate.json")
    engine = Veritas(Config(model_dir=Path(".cache/models"), page_size=3, rationale_gate=gate))
    engine.add(docs)
    start = time.perf_counter()
    engine.retrieve("scientific evidence", 3)
    build = time.perf_counter() - start
    records = []
    ungated = []
    retrieval = []
    confusion = Counter()
    for r in claims:
        result = engine.search(r["claim"], max_search_steps=1)
        if confirmatory:
            checkpoint = engine.gate
            engine.gate = None
            baseline = engine.search(r["claim"], max_search_steps=1)
            engine.gate = checkpoint
            ungated.append(baseline.claims[0].verdict)
        if r["evidence"]:
            hits = engine.retrieve(r["claim"], 10)
            retrieval.append(metrics([h.doc_id for h in hits], set(r["evidence"])))
        labels = {a["label"] for v in r["evidence"].values() for a in v}
        gold = (
            "insufficient"
            if not labels
            else "conflict"
            if len(labels) > 1
            else {"SUPPORT": "supported", "CONTRADICT": "refuted"}[labels.pop()]
        )
        pred = result.claims[0]
        correct_cites = 0
        for i in pred.citations:
            e = result.evidence[i]
            sentences = engine.docs[e.doc_id].text.splitlines()
            rationales = r["evidence"].get(e.doc_id, [])
            if any(
                e.passage in sentences[j]
                for a in rationales
                if a["label"] == {"support": "SUPPORT", "contradict": "CONTRADICT"}[e.stance]
                for j in a["sentences"]
            ):
                correct_cites += 1
        records.append(
            dict(
                claim_id=r["id"],
                gold=gold,
                predicted=pred.verdict,
                citations=len(pred.citations),
                rationale_matches=correct_cites,
                latency_ms=result.latency_ms,
            )
        )
        confusion[f"{gold}/{pred.verdict}"] += 1
    answered = [r for r in records if r["predicted"] in {"supported", "refuted"}]
    cites = sum(r["citations"] for r in records)
    correct = sum(r["rationale_matches"] for r in records)
    write(
        "results/neural_gated.json" if confirmatory else "results/neural_scifact.json",
        dict(
            seed=41 if confirmatory else 17,
            same_sample_ungated_correct=sum(v == r["gold"] for v, r in zip(ungated, records))
            if confirmatory
            else None,
            same_sample_ungated_answered=sum(v in {"supported", "refuted"} for v in ungated)
            if confirmatory
            else None,
            documents=len(docs),
            sample_size=len(claims),
            answered=len(answered),
            correct=sum(r["predicted"] == r["gold"] for r in records),
            correct_answered=sum(r["predicted"] == r["gold"] for r in answered),
            citation_precision=correct / cites if cites else None,
            matched_citations=correct,
            citations=cites,
            recall_at_10=float(np.array(retrieval)[:, 0].mean()),
            retrieval_queries=len(retrieval),
            index_and_warmup_seconds=build,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            latency_ms=np.percentile([r["latency_ms"] for r in records], [50, 95, 99]).tolist(),
            confusion=dict(confusion),
            records=records,
        ),
    )


if __name__ == "__main__":
    import sys

    main("--confirmatory" in sys.argv)
