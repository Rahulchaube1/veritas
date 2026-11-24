"""Replay the published 80 SciFact queries and retain paired, claim-level outcomes.

Run from the repository root after the neural reproduction setup in docs/reproduce.md.
"""

import json
from collections import Counter
from pathlib import Path

from scipy.stats import binomtest

from veritas import Config, Document, Veritas

ROOT = Path(".cache/scifact")


def load_rows(filename):
    return [json.loads(line) for line in (ROOT / filename).read_text().splitlines()]


def score(engine, claim, result):
    decision = result.claims[0]
    matched = 0
    for i in decision.citations:
        e = result.evidence[i]
        label = {"support": "SUPPORT", "contradict": "CONTRADICT"}[e.stance]
        sentences = engine.docs[e.doc_id].text.splitlines()
        matched += any(
            e.passage in sentences[j]
            for a in claim["evidence"].get(e.doc_id, [])
            if a["label"] == label
            for j in a["sentences"]
        )
    return {"verdict": decision.verdict, "citations": len(decision.citations), "matched": matched}


def main():
    old = json.loads(Path("results/neural_gated.json").read_text())["records"]
    dev = {r["id"]: r for r in load_rows("claims_dev.jsonl")}
    engine = Veritas(Config(model_dir=Path(".cache/models"), page_size=3))
    engine.add(
        Document(id=str(r["doc_id"]), title=r["title"], text="\n".join(r["abstract"]))
        for r in load_rows("corpus.jsonl")
    )
    gate = json.loads(Path("models/rationale_gate.json").read_text())
    rows = []
    for original in old:
        q = dev[original["claim_id"]]
        row = {"id": q["id"], "gold": original["gold"]}
        for mode, checkpoint in [("gated", gate), ("ungated", None)]:
            engine.gate = checkpoint
            row[mode] = score(engine, q, engine.search(q["claim"], max_search_steps=1))
        row["matches_archived_gated_record"] = row["gated"] == {
            "verdict": original["predicted"],
            "citations": original["citations"],
            "matched": original["rationale_matches"],
        }
        rows.append(row)
    engine.close()
    pairs = Counter((r["gated"]["verdict"] == r["gold"], r["ungated"]["verdict"] == r["gold"]) for r in rows)
    gains, losses = pairs[True, False], pairs[False, True]
    outputs = {
        "n": len(rows),
        "replayed_archive_matches": sum(r["matches_archived_gated_record"] for r in rows),
        "gains": gains,
        "losses": losses,
        "both_correct": pairs[True, True],
        "both_incorrect": pairs[False, False],
        "exact_paired_two_sided_p": binomtest(gains, gains + losses).pvalue if gains + losses else 1,
        "records": rows,
    }
    Path("paper/paired_audit.json").write_text(json.dumps(outputs, indent=2) + "\n")


if __name__ == "__main__":
    main()
