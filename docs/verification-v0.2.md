# VERITAS v0.2 verification study

Rahul Chaube · Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India · RC3649@srmist.edu.in

The release addresses three questions. Can typed evidence cross a language or document boundary? Can an independent report reach the verifier when the highest-ranked pages repeat each other? Does scoring individual passages improve scientific verdicts and citations under the same retrieval budget? Each question has its own comparison.

## Reading a verdict

A verdict describes what the indexed passages establish. Consider two worlds with the same register saying that a ferry carries 40 passengers. In the first world the register is current. In the second a recent change reduced capacity and the operator has yet to update the register. A verifier given identical records and metadata receives identical inputs in both worlds. Any definite answer about actual capacity must be wrong in at least one of them. Abstention can avoid that error at the cost of leaving the question open. Additional independent observations can distinguish the worlds. The synthetic generator deliberately includes cases where every available report agrees and the simulator truth differs.

### What can be proved about a duplicate?

For a fixed retrieved set, let each group contribute its largest eligible support strength and its largest eligible contradiction strength. Sum those maxima over groups to obtain S and C. Suppose a copy enters its existing group, does not change any previous grouping or time state, and its strength on each side is at most that group’s existing maximum. Taking a maximum with a smaller or equal value leaves that maximum unchanged. Every other group keeps its previous contribution. Therefore S, C, the reported balance (S − C)/(1 + S + C), and the verdict computed from S and C are unchanged. This is a conditional guarantee about aggregation. Retrieval can violate its assumptions by dropping a document. The crowding experiment below tests that boundary. [Mathematical specification](../research/mathematical_foundations.md)

## Same 360 synthetic cases

| Outcome | v0.1.0 | v0.2.0 |
|---|---:|---:|
| Matches the generator’s evidence verdict | 300/360 (83.33%) | 330/360 (91.67%) |
| Decisive answers | 150/360 (41.67%) | 180/360 (50.00%) |
| English claim with a Spanish station count | 0/30 | 30/30 |
| Original two-document depot claim | 0/30 | 0/30 |

The 360 test records come from twelve known templates with 30 instances each. Development of the new grammar used knowledge of the two failing families. This is a regression comparison on the original inputs. The 30 additional correct verdicts all came from the Spanish count family. The new system agreed with the hidden simulator truth on 150 of its 180 decisive answers. Thirty single-source reports asserted the wrong count. [Baseline](../results/controlled.json) · [Revised](../results/controlled_v2.json) · [Generator](../benchmarks/worlds.py)

The quantity rule parses a site type, name, unit and integer. For example, “El hospital Mar tiene 18 camas” can support “Hospital Mar has 18 beds.” “Hospital Mar has 18 seats” measures a different property. The grammar has a finite bilingual vocabulary. Name matching preserves accents and does not resolve aliases. General translation remains untested.

The two-document rule binds a relation such as “Cedar Trust operates Hospital Mar” to a count for that same hospital. The source spans for both steps accompany a decisive answer. A missing binding, a disputed binding, two matching hospitals, or an ineligible date prevents that derivation. A direct assertion can still supply independent evidence. The original two-hop fixtures call the facility a **depot** while their premises identify a **station**. None says the two types refer to the same facility. VERITAS abstained in all 30. In a separate diagnostic, changing only the requested type to **station** produced 30/30 supported verdicts with both documents cited. Those cases are outside the 360-case score. [Join implementation](../src/veritas/joins.py) · [Regression cases](../tests/test_grounded_counts.py)

## Independent evidence in a crowded ranking

The experiment crosses one, two, four, eight and sixteen copies with budgets of one, two and five steps. Each step can inspect at most two documents. The expected verdict is conflict because both the primary register and the opposing report exist in the corpus. With ordinary ranking, the verifier found the conflict in 7 of 15 settings. Prioritizing unseen normalized document texts within a wider ranked prefix found it in all 15. With 16 copies and a one-step budget, each method inspected two documents. Their verdicts were refuted and conflict respectively.

The wider search examines a candidate prefix of at least 32 before allocating the passage-verification budget. It groups repeats by normalized full text and temporal eligibility for this scheduling decision. It still spends ranking work on the wider prefix. A primary report outside it can remain invisible. The repository’s default YAML enables the option. The in-memory SDK enables it through `Config(diverse_candidates=True)`. [Ordinary ranking](../results/candidate_crowding.json) · [Diverse selection](../results/candidate_crowding_diverse.json)

## Paired SciFact comparison

We evaluated the remaining 172 claims from the public SciFact development split after excluding the earlier 48-query diagnostic and 80-query paired audit. The same 5,183 abstracts, three-document budget and single search round applied to both profiles. The comparison script checks claim IDs, gold verdicts and retrieved document IDs before computing paired statistics. Each profile examined three documents for 170 claims and skipped retrieval for two inputs rejected by the query parser. SciFact’s development data had already contributed to earlier aggregate experiments. There is no private-test estimate here. [SciFact paper](https://aclanthology.org/2020.emnlp-main.609/) · [Earlier 80-query audit](../paper/paired_audit.json)

The baseline uses the v0.1.0 rationale-gated model. The revised scientific profile retains document retrieval, considers up to twelve lexical passage candidates per document, and applies the MS MARCO cross-encoder to unresolved sentences. A passage passes if its learned lexical gate reaches the stored threshold, or if that score reaches 0.05 and the reranker’s sigmoid score reaches 0.5. The sum of the two scores ranks eligible sentences. At most two per document reach the quantized DeBERTa model trained on MNLI, FEVER and ANLI. The stance threshold is 0.90. These numerical thresholds control the decision rule and do not estimate the probability that a scientific claim is true. This comparison changes several components together and cannot assign the effect to any one of them. [Profile](../configs/scientific.yaml) · [Passage selector](../src/veritas/grounding.py) · [Pinned model sources](../scripts/fetch_models.py)

| Measure | v0.1.0 gated | v0.2.0 scientific |
|---|---:|---:|
| Verdict matches gold, including insufficient | 84/172 (48.8%) | 93/172 (54.1%) |
| Decisive verdicts | 33/172 (19.2%) | 53/172 (30.8%) |
| Correct among decisive answers | 22/33 (66.7%) | 34/53 (64.2%) |
| Incorrect decisive answers | 11 | 19 |
| Correct answers with at least one matching citation | 9/172 (5.2%) | 18/172 (10.5%) |
| Matching citations among all decisive citations | 9/37 (24.3%) | 18/62 (29.0%) |

“Matching citation” means the cited span lies in a gold rationale sentence for the right document and stance. The metric can miss valid unannotated evidence. A correct verdict on an insufficient-evidence case counts in the first row. It cannot count as a grounded decisive answer. Per-query records include verdicts, retrieved document IDs, citation offsets into the newline-joined abstract, zero-based indices of matching gold rationale sentences, and observed latency. The runs used a Linux x86-64 CPU host (AMD EPYC 9V74) with CPU ONNX inference. The two processes shared compute resources. The recorded latency percentiles describe those runs and cannot establish an isolated speed comparison.

On paired verdict correctness, 23 cases improved and 14 regressed. The two-sided exact binomial test on the 37 discordant pairs gives p = 0.188. The sample does not establish a population-wide gain at the conventional 0.05 threshold. The revised system produced 12 more correct decisive answers and eight more incorrect decisive answers. The scientific profile is optional. Independent scientific review and evaluation on a new split remain necessary for decisions that rely on factual correctness. [Paired summary](../results/scientific_172_summary.json) · [Baseline records](../results/scientific_172_baseline.json) · [Revised records](../results/scientific_172_revised.json)

## Reproduce the comparison

Run from the repository root. Use the standard setup instructions and download the official corpus and weights. The baseline command loads the implementation from the v0.1.0 tag while using the current evaluation script. Both profiles read the same stored rationale gate. The summary command alone replays the comparison from the committed per-query files. The audit command checks each citation offset and rationale match against the official archive without loading any models. [Environment and setup](reproduce.md)

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=17
python scripts/fetch_scifact.py
python scripts/fetch_models.py dense rerank nli scientific_nli
git worktree add --detach .cache/veritas-v010 v0.1.0
PYTHONPATH="$PWD/.cache/veritas-v010/src" python experiments/compare_scientific.py baseline
PYTHONPATH="$PWD/src" python experiments/compare_scientific.py revised
python experiments/compare_scientific.py summary
python experiments/compare_scientific.py audit
python experiments/controlled.py --output-root .cache/controlled-replay
PYTHONPATH=experiments python -c 'from extractive import crowding; crowding(diverse=True)'
```
