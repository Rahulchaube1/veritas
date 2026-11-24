# Collections and annotation guide

**VERITAS-Bench v0.1** contains 1,440 machine-generated closed worlds, 120 per family. Each record includes the query, documents, source-family identifiers, applicable date, expected evidence verdict and independently specified simulator truth when defined. There are 720 development cases, 360 calibration cases and 360 test cases. Identifiers are disjoint. Templates are deliberately shared across partitions. Results characterize these controlled perturbations and have a much smaller effective diversity than 1,440 independently authored claims. The fixture is distinct from the ACL VeriTaS benchmark. [Generator](../benchmarks/worlds.py) · [ACL benchmark](https://aclanthology.org/2026.acl-long.1948/)

The twelve families test copied errors, independent reports, citation-linked syndication, conflict, expired evidence, false premises, insufficient evidence, numerical disagreement, Spanish evidence, two-hop claims, unsupported reports and embedded instructions. The unsupported-report case places a false statement in the corpus while the synthetic world records the opposite. It represents machine-authored misinformation through a template. It contains no sampled language-model output. The instruction family uses inert plain text. No examples carry human-verified labels.

For human review, identify the subject, predicate, value, unit and date. Record exact text offsets for each supporting or contradicting passage. Record explicit citation edges separately from suspected copying. Use conflict when incompatible eligible observations compete, and insufficient when the corpus fails to justify either polarity. An independent annotator should label the same material before adjudication. This release has no inter-annotator agreement measurement.

**SciFact.** The downloader fetches the official Allen Institute for AI archive, verifies its SHA-256 and reads three named members. The original claims and annotations are CC BY 4.0. The upstream release attributes abstracts to S2ORC under ODC-By 1.0. This repository publishes aggregate results and claim IDs. Obtain the underlying documents from the official release. Credit Wadden et al. and the original abstract authors. [Release and license](https://github.com/allenai/scifact/blob/master/LICENSE.md) · [Paper](https://arxiv.org/abs/2004.14974)

**Public quick start.** Two short excerpts are attributed individually to NASA and the U.S. National Archives. The Seventeenth Amendment ratification date supplies the example’s lower validity bound. U.S. federal works and third-party content have different reuse rules. These excerpts contain no agency logos or endorsement. [NASA](https://science.nasa.gov/neptune/facts/) · [National Archives](https://www.archives.gov/founding-docs/amendments-11-27)

The original synthetic records in this repository are released under CC0-1.0. The source-code MIT license covers the generator. Third-party content and checkpoints retain their own terms. [CC0](https://creativecommons.org/publicdomain/zero/1.0/)

## Other benchmark coverage

| Collection | What it would test | Release status |
|---|---|---|
| BEIR | Ranked document relevance | BEIR-format corpus, query and qrels adapter supplied. Native SciFact is the measured corpus. [BEIR](https://github.com/beir-cellar/beir) |
| FEVER and FEVEROUS | Wikipedia evidence and table cells | Candidate datasets reviewed. Table-cell grounding and Wikipedia snapshot evaluation are outstanding. [FEVER](https://fever.ai/dataset/fever.html) · [FEVEROUS](https://fever.ai/dataset/feverous.html) |
| FreshQA | Changing facts and false premises | Synthetic interval checks executed. FreshQA needs an aligned, dated evidence collection. [FreshLLMs](https://github.com/freshllms/freshqa) |
| LongFact | Long-form factuality | Local extractive proxy evaluated separately. No official SAFE or LongFact score. [LongFact](https://github.com/google-deepmind/long-form-factuality) |
| ALCE | Attribution of generated sentences | SciFact rationale matching is measured. ALCE’s multi-dataset long-form protocol remains unevaluated. [ALCE](https://github.com/princeton-nlp/ALCE) |
| FACTS Grounding and suite | Grounding, search, parametric and multimodal factuality | Protocols reviewed. No official judge, private labels or leaderboard submission used. [Suite](https://deepmind.google/blog/facts-benchmark-suite-systematically-evaluating-the-factuality-of-large-language-models/) |
