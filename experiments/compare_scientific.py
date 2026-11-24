"""Paired evaluation on all SciFact dev IDs outside the two previous neural audits.

Run each variant with the designated source tree first on PYTHONPATH.
Only gold labels used for scoring are read after each query has been answered.
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rationale_spans import matching_sentences
from scifact import corpus, rows, write
from scipy.stats import binomtest

from veritas import Config, Veritas


def evaluate(variant: str, out: Path):
    used = {
        r["claim_id"]
        for filename in ["neural_scifact", "neural_gated"]
        for r in json.loads(Path(f"results/{filename}.json").read_text())["records"]
    }
    queries = [r for r in rows("claims_dev") if r["id"] not in used]
    if len(queries) != 172 or len({r["id"] for r in queries}) != 172:
        raise ValueError("Expected 172 distinct remaining SciFact development claims")
    config = {
        "model_dir": Path(".cache/models"),
        "rationale_gate": Path("models/rationale_gate.json"),
        "page_size": 3,
    }
    if variant == "revised":
        config["scientific_evidence"] = True
    engine = Veritas(Config.model_validate(config))
    engine.add(corpus())
    start = time.perf_counter()
    engine.retrieve("scientific evidence", 3)
    startup = time.perf_counter() - start
    records = []
    for row in queries:
        result = engine.search(row["claim"], max_search_steps=1)
        decision = result.claims[0]
        refs = row["evidence"]
        labels = {span["label"] for spans in refs.values() for span in spans}
        gold = (
            "insufficient"
            if not labels
            else "conflict"
            if len(labels) > 1
            else {"SUPPORT": "supported", "CONTRADICT": "refuted"}[labels.pop()]
        )
        citation_spans = []
        for idx in decision.citations:
            e = result.evidence[idx]
            text = engine.docs[e.doc_id].text
            wanted = {"support": "SUPPORT", "contradict": "CONTRADICT"}[e.stance]
            eligible = {
                j for ref in refs.get(e.doc_id, []) if ref["label"] == wanted for j in ref["sentences"]
            }
            gold_sentences = matching_sentences(text, e.start, e.end, eligible)
            if engine.docs[e.doc_id].text[e.start : e.end] != e.passage:
                raise ValueError(f"Invalid citation offsets for claim {row['id']}")
            citation_spans.append(
                {
                    "doc_id": e.doc_id,
                    "start": e.start,
                    "end": e.end,
                    "stance": e.stance,
                    "gold_sentence_matches": gold_sentences,
                }
            )
        records.append(
            {
                "id": row["id"],
                "verdict": decision.verdict,
                "gold": gold,
                "citations": len(decision.citations),
                "matched": sum(bool(span["gold_sentence_matches"]) for span in citation_spans),
                "citation_spans": citation_spans,
                "latency_ms": result.latency_ms,
                "examined": result.documents_examined,
                "doc_ids": sorted({e.doc_id for e in result.evidence}),
            }
        )
    engine.close()
    write(out, {"variant": variant, "index_warmup_seconds": startup, "records": records})


def summarize(baseline: Path, revised: Path):
    samples = {
        name: json.loads(path.read_text()) for name, path in [("baseline", baseline), ("revised", revised)]
    }
    old, new = [block["records"] for block in samples.values()]
    if not old or [(r["id"], r["gold"], r["doc_ids"]) for r in old] != [
        (r["id"], r["gold"], r["doc_ids"]) for r in new
    ]:
        raise ValueError("Mismatched evaluation cases or retrieved documents")
    result = {}
    for name, block in samples.items():
        records = block["records"]
        decisive = [r for r in records if r["verdict"] in {"supported", "refuted"}]
        good = [r for r in decisive if r["verdict"] == r["gold"]]
        result[name] = {
            "queries": len(records),
            "correct_all": sum(r["verdict"] == r["gold"] for r in records),
            "answered": len(decisive),
            "correct_decisive": len(good),
            "wrong_decisive": len(decisive) - len(good),
            "grounded_correct": sum(r["matched"] > 0 for r in good),
            "citations": sum(r["citations"] for r in records),
            "matched_citations": sum(r["matched"] for r in records),
            "latency_ms_p50_p95": np.percentile([r["latency_ms"] for r in records], [50, 95]).tolist(),
        }
    pairs = Counter((a["gold"] == a["verdict"], b["gold"] == b["verdict"]) for a, b in zip(old, new))
    wins, losses = pairs[False, True], pairs[True, False]
    result["paired_correctness"] = {
        "gains": wins,
        "losses": losses,
        "exact_two_sided_p": binomtest(wins, wins + losses).pvalue if wins + losses else 1.0,
    }
    return result


def audit_spans(path: Path):
    """Replay the published citation offsets against the official corpus without neural weights."""
    docs = {doc.id: doc for doc in corpus()}
    gold = {r["id"]: r["evidence"] for r in rows("claims_dev")}
    sample = json.loads(path.read_text())
    checked = matched = 0
    for record in sample["records"]:
        cites = record["citation_spans"]
        if record["citations"] != len(cites):
            raise ValueError(f"Citation count differs for {record['id']}")
        aligned = 0
        for cite in cites:
            doc_id = cite["doc_id"]
            text = docs[doc_id].text
            start, end = cite["start"], cite["end"]
            if doc_id not in record["doc_ids"] or not 0 <= start < end <= len(text):
                raise ValueError(f"Invalid source span for {record['id']}")
            stance = {"support": "SUPPORT", "contradict": "CONTRADICT"}[cite["stance"]]
            eligible = {
                i for a in gold[record["id"]].get(doc_id, []) if a["label"] == stance for i in a["sentences"]
            }
            aligned_lines = matching_sentences(text, start, end, eligible)
            if cite["gold_sentence_matches"] != aligned_lines:
                raise ValueError(f"Citation alignment differs for {record['id']}")
            aligned += bool(aligned_lines)
        if aligned != record["matched"]:
            raise ValueError(f"Gold match count differs for {record['id']}")
        matched += aligned
        checked += len(cites)
    return {"variant": sample["variant"], "checked": checked, "matched": matched}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["baseline", "revised", "summary", "audit"])
    args = p.parse_args()

    def path(name):
        return Path(f"results/scientific_172_{name}.json")

    if args.action == "summary":
        write(path("summary"), summarize(path("baseline"), path("revised")))
    elif args.action == "audit":
        print(json.dumps([audit_spans(path(name)) for name in ["baseline", "revised"]], indent=2))
    else:
        evaluate(args.action, path(args.action))
