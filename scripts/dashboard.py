import json
from pathlib import Path

r = json.loads(Path("results/scifact.json").read_text())
c = json.loads(Path("results/controlled.json").read_text())
rows = [
    [
        label,
        format(r["results"][k]["recall@10"]["mean"], ".3f"),
        format(r["results"][k]["ndcg@10"]["mean"], ".3f"),
    ]
    for k, label in [
        ("bm25", "BM25"),
        ("dense", "Character LSA"),
        ("hybrid", "Rank fusion"),
        ("learned_hybrid", "Fusion + learned relevance"),
    ]
]
Path("src/veritas/web/benchmark.json").write_text(
    json.dumps(
        dict(
            description=f"SciFact · {r['documents']:,} abstracts · {r['retrieval_queries']} evidence-labeled development claims",
            rows=rows,
            controlled=f"On {c['worlds']} synthetic test cases, the full rule pipeline matched {c['ablations']['full']['verdict_accuracy']:.1%} of evidence labels. The rules abstained on all 300 real SciFact claims.",
        ),
        indent=2,
    )
    + "\n"
)
