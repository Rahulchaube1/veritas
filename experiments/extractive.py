"""Equal-corpus extractive answer baselines and a separate-search corroboration proxy."""

from scifact import corpus, rows, write

from veritas import Config, Veritas
from veritas.provenance import graph
from veritas.retrieval import tokens
from veritas.verification import passages, stance


def main():
    docs = corpus()
    dev = rows("claims_dev")
    engine = Veritas()
    engine.add(docs)
    results = {}
    for mode in ["bm25", "dense", "hybrid", "hybrid_verified", "separate_search"]:
        emitted = aligned = calls = 0
        for row in dev:
            q = row["claim"]
            qset = set(tokens(q))
            hits = engine.retrieve(q, 4, mode if mode in {"bm25", "dense"} else "hybrid")
            calls += 1
            if not hits:
                continue
            choices = [
                (len(qset & set(tokens(p))) / max(1, len(qset | set(tokens(p)))), d, p)
                for hit in hits
                for d in [engine.docs[hit.doc_id]]
                for p, _, _ in passages(d)
            ]
            if not choices:
                continue
            _, doc, p = max(choices, key=lambda v: v[0])
            if mode == "hybrid_verified" and stance(q, p)[0] == "neutral":
                continue
            if mode == "separate_search":
                # SAFE-inspired extra search on an extracted statement, local corpus only.
                new = engine.retrieve(p[:1600], 4)
                calls += 1
                peers = [engine.docs[h.doc_id] for h in new if h.doc_id != doc.id]
                groups, _ = graph([doc, *peers])
                if not any(
                    groups[d.id] != groups[doc.id]
                    and any(stance(p, s)[0] == "support" for s, _, _ in passages(d))
                    for d in peers
                ):
                    continue
            emitted += 1
            sentences = doc.text.splitlines()
            aligned += any(p in sentences[j] for a in row["evidence"].get(doc.id, []) for j in a["sentences"])
        results[mode] = dict(
            queries=len(dev),
            emitted=emitted,
            selected_gold_passages=aligned,
            rationale_alignment=aligned / emitted if emitted else None,
            retrieval_calls=calls,
        )
    write("results/extractive_baselines.json", results)


def crowding(diverse=False):
    records = []
    for n in [1, 2, 4, 8, 16]:
        for budget in [1, 2, 5]:
            e = Veritas(Config(page_size=2, diverse_candidates=diverse))
            claim = "The ferry carries 40 passengers."
            e.add(
                [dict(id="primary", text=claim, title="Survey record")]
                + [
                    dict(
                        id=f"copy{j}",
                        title="The ferry carries 40 passengers.",
                        text="The ferry carries 41 passengers.",
                    )
                    for j in range(n)
                ]
            )
            r = e.search(claim, budget)
            records.append(
                dict(
                    copies=n,
                    budget=budget,
                    verdict=r.claims[0].verdict,
                    examined=r.documents_examined,
                    steps=r.search_budget_used,
                )
            )
            e.close()
    name = "candidate_crowding_diverse" if diverse else "candidate_crowding"
    write(f"results/{name}.json", records)


if __name__ == "__main__":
    main()
    crowding()
    crowding(diverse=True)
