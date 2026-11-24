# Reproduce the release

Clone this repository, enter its root and use Python 3.12 for the recorded configuration. Direct package versions from the measured environment are in configs/constraints.txt. Timings depend on hardware and system load. The scripts overwrite their named measurement files when rerun.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -c configs/constraints.txt -e '.[dev,neural]' bandit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=17
python scripts/fetch_scifact.py
python scripts/fetch_models.py dense nli rerank
python experiments/scifact.py
python experiments/grounding.py
python experiments/controlled.py
python experiments/extractive.py
python experiments/neural_scifact.py
python experiments/neural_scifact.py --confirmatory
python scripts/dashboard.py
ruff check .
ruff format --check .
mypy src/veritas
pytest --cov=veritas
bandit -q -r src/veritas -lll
python -m build
docker build -t veritas-local:0.1 .
```

The container build accepts an optional BuildKit secret named build_ca for deployments using a private package-registry trust root. The release was built using the local system CA bundle supplied through that temporary mount. Supply a private, randomly generated VERITAS_TOKEN of at least 24 characters to Docker Compose. The compose file binds the service to loopback. The indexed sample documents persist in its named volume.

For neural search:

```bash
veritas --config configs/neural_gated.yaml index datasets/public_examples.jsonl
veritas --config configs/neural_gated.yaml search "Neptune is the eighth and most distant planet in our solar system."
veritas --config configs/neural_gated.yaml serve
```

For SearXNG, set VERITAS_SEARXNG to your authorized HTTPS JSON search endpoint, then run `veritas import-searxng "query"`. The command imports excerpts only. It requires an operator-supplied endpoint. It is disabled in the HTTP API. For BEIR, obtain a licensed collection in standard layout, then run `python scripts/evaluate_beir.py /path/to/collection`. The recorded NFCorpus run can be reproduced with `pip install pyarrow`, `python scripts/fetch_nfcorpus.py`, then `python scripts/evaluate_beir.py .cache/nfcorpus`. The downloader pins the BEIR maintainers’ two repository revisions. The adapter reads graded qrels and uses the same corpus and document limits as the research engine.

## Data and decisions

* SciFact training excludes normalized claims and exact document text overlapping official-dev evidence sources as specified by the training scripts. Logistic fits use source-text hashes to partition annotated examples. The ranker’s calibration partition is reserved for a diagnostic loss. The learned scientific passage gate uses a disjoint validation partition for threshold selection.
* The public retrieval test covers all 188 SciFact development queries with evidence annotations. The full rule-verification evaluation includes all 300 development claims. Official private test labels are unavailable. Abstracts without gold judgments may still be relevant.
* The first 48-claim neural diagnostic used seed 17. Its poor citation grounding motivated a rationale gate. A further sample of 80, seed 41, excludes those 48 IDs. Both neural variants ran on the same 80 queries, with the same three-document budget. All 300 had already contributed to earlier aggregate rule/retrieval metrics. This is a held-aside **neural comparison within public development data**, without a private, untouched test set.
* One thousand five hundred query bootstrap replicates give the retrieval intervals. Templates repeat across the synthetic splits, so per-row confidence intervals would imply a misleading degree of independence. Neural samples are small and use one split. Gold citation alignment counts a cited passage if it is contained in a gold rationale sentence from the correct document with the predicted stance. Unannotated but valid passages may be counted as misses.
* The quick-start code and Colab notebook share the actual package implementation. Training code rebuilds the small fitted heads from scratch. Exported artifacts allow inference without training.

## Inspecting results

`results/scifact.json` contains lexical, latent, fused and learned retrieval metrics and bootstrap intervals. `results/controlled.json` contains paired component ablations, duplicate injection, synthetic calibration and latency. `results/candidate_crowding.json` varies the allowed prefix independently of copy count. `results/extractive_baselines.json` records gold passage alignment for three extractive generators, a strict verification filter and a separate-search proxy inspired by SAFE. The proxy has a different search cost and does not implement the official SAFE judge. Neural files contain per-query predictions and direct counts. [Measurements](../results/)
