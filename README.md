# VERITAS

**Verified Evidence Retrieval with Independent Temporal Attribution and Scoring** checks a factual claim against a local document collection. It returns the passages it examined, identifies reports that may share a source, checks their validity dates, and explains whether the indexed evidence supports the claim, refutes it, conflicts, or is insufficient.

**Author and project founder**<br>
Rahul Chaube<br>
Department of Computing Technologies<br>
SRM Institute of Science and Technology<br>
Kattankulathur, Tamil Nadu, India<br>
[RC3649@srmist.edu.in](mailto:RC3649@srmist.edu.in)

[Read the expanded paper (PDF)](paper/veritas.pdf) · [Run in Google Colab](https://colab.research.google.com/github/Rahulchaube1/veritas/blob/main/notebooks/veritas.ipynb) · [Download the software](https://github.com/Rahulchaube1/veritas/releases/tag/v0.1.0) · [Reproduce the results](docs/reproduce.md)

## The problem in one example

A harbor register says a ferry carries **40 passengers**. An operator notice says **41**, and eight digests copy that notice. Counting pages gives nine votes to one side. A reviewer still has two competing observations. VERITAS groups dependent documents and caps each group’s contribution to each side. If an old register is present, the requested date also determines whether that register can vote.

There is a limit to what grouping can recover. In the recorded experiment with sixteen opposing copies, a ten-document retrieval budget excluded the primary observation. The verifier returned a refutation because it never saw the conflict. The paper gives both the conditional result and this counterexample. [Copy-count experiment](results/controlled.json) · [Candidate-budget experiment](results/candidate_crowding.json)

## What is original here?

VERITAS is an **independently designed and implemented verification algorithm**. Its contribution is the particular decision rule combining time-eligible passages, separate supporting and opposing maxima for each source group, bounded retrieval, and auditable citations. Its evaluation varies copying and candidate visibility separately. That distinction makes it possible to identify when aggregation works and when retrieval has already lost necessary evidence.

BM25, reciprocal rank fusion, source-dependency modeling, NLI and isotonic calibration all have prior art. The paper establishes a conditional invariance property for this composition. It makes no claim of worldwide priority or superior general-purpose search accuracy. An unrelated ACL benchmark also uses the name *VeriTaS*. Use the repository and version when citing this project. [Prior work](research/literature_review.md) · [Novelty assessment](research/novelty_assessment.md) · [ACL benchmark](https://aclanthology.org/2026.acl-long.1948/)

## Run your first verification

Use Python 3.11 or later. The recorded experiments used Python 3.12 on CPU. Run these commands in a shell with Git and Python available.

```bash
git clone https://github.com/Rahulchaube1/veritas.git
cd veritas
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
veritas index datasets/public_examples.jsonl
veritas search "Neptune is the eighth and most distant planet in our solar system."
```

The sample collection includes an attributed NASA passage and an excerpt from the U.S. National Archives. For the Neptune query, look for `supported` in `claims[0].verdict`, then follow its first citation index into `evidence` to read the source passage and URL. Factual confidence is null in this configuration. All search computation runs locally after installation.

To use the browser interface, run `veritas serve` and open [the local search page](http://127.0.0.1:8000). The same service exposes [interactive API documentation](http://127.0.0.1:8000/docs). The Colab notebook runs the repository implementation and includes the copy experiment. [Notebook](notebooks/veritas.ipynb)

## Index your own collection

Provide one JSON document per line to `veritas index your-documents.jsonl`. Every record needs an `id` and `text`, or an `id` and `html` for local plain-text extraction. Reusing an ID replaces that document. The next retrieval rebuilds the in-memory index. The default repository configuration persists the documents in SQLite.

| Field | What to supply |
|---|---|
| `id`, `title`, `text` | Stable document ID, optional display title, and source text. Citations refer to offsets in the stored text. |
| `publisher`, `author`, `url` | Known attribution and an optional HTTP(S) citation URL. Publisher identity can affect grouping. |
| `cites` | Referenced identifiers or URLs. Indexed document IDs can also be followed in deep mode. |
| `published`, `modified` | Dates for the version available to the reader. |
| `valid_from`, `valid_to` | Inclusive boundaries for the fact asserted in this document. The engine uses supplied boundaries for time-scoped verification. |
| `event_date` | Event metadata. An event date by itself does not establish the validity interval. |

Here is a complete in-memory SDK example. Both reports are eligible on the requested date, so the expected verdict is `conflict`.

```python
from veritas import Veritas

engine = Veritas()
try:
    engine.add(
        [
            {"id": "register", "text": "The ferry carries 40 passengers.", "valid_from": "2023-01-01"},
            {"id": "notice", "text": "The ferry carries 41 passengers.", "valid_from": "2023-01-01"},
            {
                "id": "digest",
                "text": "The ferry carries 41 passengers.",
                "cites": ["notice"],
                "valid_from": "2023-01-01",
            },
        ]
    )
    result = engine.search("The ferry carries 40 passengers.", as_of="2024-06-01")
    print(result.claims[0].verdict, result.evidence_balance)  # conflict 0.0
    for passage in result.evidence:
        print(passage.doc_id, passage.stance, passage.temporal, passage.group)
finally:
    engine.close()
```

## How the algorithm reaches a verdict

| Stage | Operation | What you can inspect |
|---|---|---|
| Retrieve | Rank documents with BM25 and a vector representation. Combine ranks using reciprocal rank fusion. | Document IDs and feature-level ranking scores. |
| Select passages | Split stored text into exact spans and select up to five per document by token overlap. | Passage, character offsets and source metadata. |
| Check time and stance | Determine availability and validity at the query date. Apply exact, negation and numeric rules, with optional neural inference for unresolved passages. | Time state, support/contradict/neutral stance, method and strength. |
| Group sources | Link identical text or URLs, citations, shared references, quotes, publishers and near duplicates. | Edge type, basis, and whether it joined the documents for voting. |
| Aggregate | Keep the strongest eligible support and contradiction in each group. Sum the maxima separately. | Group assignments, signed balance and eligible contradictions. |
| Decide and cite | Spend more retrieval budget when the stopping rule permits. Return a decisive verdict only with a surviving source span. | Verdict, citations, abstention, actions, reviewed-document count and latency. |

If the summed supporting and opposing weights are `S` and `C`, the signed balance is `(S - C) / (1 + S + C)`. For example, one independent unit observation on each side gives zero and a conflict. With no eligible evidence, zero instead accompanies an insufficient verdict. Read the verdict together with the balance.

The grouping property assumes all earlier evidence remains visible, its eligibility and strength remain fixed, and the copy enters its existing group without raising the group maximum. Copies can still crowd out a primary source. A new document can also connect previously separate groups. [Algorithm specification](research/algorithm_specification.md) · [Mathematical foundations](research/mathematical_foundations.md)

## Choose an inference configuration

| Configuration | Retrieval | Verification | Appropriate use |
|---|---|---|---|
| [Default](configs/default.yaml) | BM25 and character TF–IDF/LSA | Deterministic rules | Offline indexing, reproducible traces, numeric and temporal examples. |
| [Neural](configs/neural.yaml) | Multilingual MiniLM, HNSW and an English cross-encoder | Rules plus DeBERTa NLI | CPU experiments on paraphrased assertions. |
| [Neural with passage filter](configs/neural_gated.yaml) | Same neural retrieval backend | Learned SciFact rationale filter before NLI | Studying citation quality versus how often the system answers. |

```bash
python -m pip install -e '.[neural]'
python scripts/fetch_models.py dense nli rerank
veritas --config configs/neural_gated.yaml index datasets/public_examples.jsonl
veritas --config configs/neural_gated.yaml search "Neptune is the eighth and most distant planet in our solar system."
```

The model downloader pins upstream revisions. Optional project-trained JSON files contain document relevance weights, the rationale filter, and a synthetic calibrator. That calibrator estimates agreement with simulator truth among decisive answers in the controlled benchmark. It is disabled in the standard search configurations. [Model cards](models/README.md)

## What the experiments found

| Evaluation | Measured outcome |
|---|---|
| SciFact retrieval, 5,183 abstracts and 188 evidence-labeled queries | Mean Recall@10 was **0.930** for BM25, **0.478** for character LSA and **0.864** for rank fusion. [Retrieval measurements](results/scifact.json) |
| BEIR NFCorpus, 323 test queries | The default hybrid configuration produced mean Recall@10 **0.134** and nDCG@10 **0.264**. [NFCorpus measurements](results/nfcorpus.json) |
| Synthetic evidence verdicts | **300/360** matched the generator’s evidence labels. Removing grouping or removing time eligibility separately reduced that to **270/360**. Default rules missed all cases in the multilingual and two-hop families. [Ablations](results/controlled.json) |
| Scientific verification with rules | The rule verifier abstained on **all 300** SciFact development claims. It matched the **112** gold insufficient-evidence judgments. [Full evaluation](results/scifact.json) |
| Paired neural comparison, 80 scientific claims | Without the filter, **22/53** decisive answers were correct and **6/83** citations aligned. With the filter, **11/18** decisive answers were correct and **6/19** citations aligned. [Query-level replay](paper/paired_audit.json) |

The paired replay identifies the practical trade-off. The filter reduced incorrect decisive answers from **31 to 7**. It also reduced correct decisive answers from **22 to 11**. Six queries in each configuration received a correct verdict with at least one gold-aligned citation. These public development results support further research and human review. They do not establish reliability for unattended scientific fact-checking. [Expanded paper](paper/veritas.pdf) · [All measurements](results/)

## HTTP API and response fields

| Endpoint | Purpose |
|---|---|
| `POST /v1/index/documents` | Validate and atomically upsert a batch. |
| `POST /v1/retrieve` | Retrieve ranked document IDs and score features. |
| `POST /v1/search` | Check one claim or supported question form. |
| `POST /v1/verify` | Verify sentence-separated assertions within one total budget. |
| `POST /v1/evidence/analyze` | Inspect the query’s passage stances and source relationships. |
| `GET /v1/evidence/{claim_id}` | Read the most recent cached analysis for a claim. |
| `GET /v1/health`, `/v1/models`, `/v1/metrics` | Inspect corpus size, configured models and process activity. |

The `claims[].citations` values are **zero-based indices into the response’s evidence array**. A conflict can contain evidence on both sides and have no decisive-answer citations. For a time-scoped search, check `evidence[].temporal` before treating a passage as eligible. For historical reproducibility, retain the exact indexed text along with its ID and offsets. [Validated result schema](docs/result-schema.json)

The service binds to loopback by default. The Docker configuration supports a bearer token for remote operation. Indexed URLs are displayed as citations. The optional SearXNG import uses an operator-configured endpoint. Multi-tenant permissions, load testing and real-world source-lineage validation remain open. [Operating limits](SECURITY.md) · [Development status](STATUS.md)

## Reproduce, inspect and cite

| Goal | Start here |
|---|---|
| Follow the train, validation and evaluation splits | [Reproduction guide](docs/reproduce.md) and [dataset card](datasets/README.md) |
| Recompute the 80-query paired comparison | [Replay script](paper/paired_audit.py) and [per-query records](paper/paired_audit.json) |
| Read the research argument or rebuild the paper | [Paper PDF](paper/veritas.pdf), [LaTeX source](paper/veritas.tex) and [paper build notes](paper/README.md) |
| Inspect software checks and measured limitations | [Validation record](results/validation.json) and [theoretical limitations](research/theoretical_limitations.md) |
| Cite or reuse the project | [Citation metadata](CITATION.cff), [software license](LICENSE) and [third-party dataset terms](datasets/README.md) |
