# VERITAS

Verified Evidence Retrieval with Independent Temporal Attribution and Scoring is an independent research project founded by **Rahul Chaube**. It checks a claim against a local document collection, reports which passages support or contradict it, groups potentially dependent reports and abstains when eligible evidence is insufficient.

[Run the notebook in Colab](https://colab.research.google.com/github/Rahulchaube1/veritas/blob/main/notebooks/veritas.ipynb) · [Research paper](paper/veritas.pdf) · [Measured results](results/) · [Reproduce the experiments](docs/reproduce.md)

## What the experiments found

| Experiment | Observed result |
|---|---|
| Repeating one opposing report | With all evidence in view, 1–16 copies left the grouped balance at zero. Naive voting moved toward the repeated report. |
| Retrieval over 5,183 SciFact abstracts | BM25 achieved 0.930 mean Recall@10 on 188 development claims with annotated evidence. Local character-LSA rank fusion achieved 0.864. |
| Synthetic verification | 300 of 360 evidence verdicts matched the generator labels. The rule pipeline failed the multilingual and two-hop families. Templates are shared across splits. |
| Scientific verification | The default rule verifier abstained on all 300 real claims. On a separate 80-claim neural evaluation, the trained passage filter answered 18, got 11 of those right, and matched 6 of 19 citations to an annotated rationale. |
| Candidate crowding | Sixteen copies could push the primary source outside a ten-document budget. Grouping the retrieved reports could no longer recover the missing conflict. |

These results support a narrow property of evidence aggregation. The scientific results leave substantial error in source selection and verification. General factual accuracy, independent source discovery at web scale and production readiness remain unvalidated. [Raw measurements](results/) · [Limitations](research/theoretical_limitations.md)

## Run a search

Python 3.11 or newer is required. Run the commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
veritas index datasets/public_examples.jsonl
veritas search "Neptune is the eighth and most distant planet in our solar system."
veritas serve
```

Open the local interface at `http://127.0.0.1:8000` or the OpenAPI interface at `/docs`. The sample collection contains attributed excerpts from NASA and the National Archives. The default installation needs no model download, API key or paid service. For verification of paraphrased claims, use the optional neural configuration in the reproduction guide.

```python
from veritas import Veritas

engine = Veritas.from_config("configs/default.yaml")
result = engine.search("Neptune is the eighth and most distant planet in our solar system.")
print(result.answer)
for i in result.claims[0].citations:
    print(result.evidence[i].passage, result.evidence[i].url)
engine.close()
```

## Interfaces

| Endpoint | Operation |
|---|---|
| POST /v1/search | Evidence verdict for one claim or supported question form |
| POST /v1/verify | Check sentence-separated assertions within one total step budget |
| POST /v1/retrieve | Ranked documents and score breakdowns |
| POST /v1/index/documents | Atomic document upserts |
| POST /v1/evidence/analyze | Inspect passage-level stance and provenance for a query |
| GET /v1/evidence/{claim_id} | Most recent cached analysis for a claim |
| GET /v1/health, /v1/models, /v1/metrics | Corpus size, backend configuration and process counts |

Confidence defaults to null. Signed evidence balance, stance strength, retrieval relevance and factual correctness are different quantities. The exported calibrator applies only to the synthetic benchmark population. The UI renders document text as data and exposes model and temporal decisions with each passage. Deep mode can follow indexed citation identifiers and expand the next retrieval using title terms.

## Research materials

* [Literature review](research/literature_review.md), [related work](research/related_work.md) and [novelty assessment](research/novelty_assessment.md)
* [Mathematical foundations](research/mathematical_foundations.md), [algorithm](research/algorithm_specification.md) and [complexity](research/complexity_analysis.md)
* [Dataset card and licenses](datasets/README.md), [model cards](models/README.md) and [development status](STATUS.md)
* [Operating limits](SECURITY.md) and [reproducibility guide](docs/reproduce.md)

This project shares a short name with an unrelated ACL benchmark, *Verified Theses and Statements*. Use the repository name and version when citing this work. [ACL paper](https://aclanthology.org/2026.acl-long.1948/)
