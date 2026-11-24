# arXiv submission materials

**Upload** [LaTeX source archive](veritas-arxiv.tar.gz?raw=true). It contains the manuscript, three vector figures, a local bibliography, the matching generated BBL and an arXiv compiler directive for XeLaTeX. [Reference PDF](../veritas.pdf)

**Title**

Beyond Relevance: Dependency-Aware and Temporally Grounded Evidence Verification for Reliable Generative Search

**Author**

Rahul Chaube<br>
Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India<br>
RC3649@srmist.edu.in

**Suggested primary category**

cs.IR (Information Retrieval)

**Abstract for the submission form**

Counting pages overstates corroboration when several publishers repeat one report. We study how an evidence verifier behaves when the number of copies changes while the number of observations stays fixed. VERITAS retrieves passages, checks temporal eligibility, groups documents by observable dependency signals, and sums the strongest supporting and opposing evidence in each group. On a controlled collection, increasing the number of opposing copies from one to sixteen left the grouped balance unchanged. Under a ten-document budget the copies could exclude the primary source, turning an observable conflict into a refutation. On 188 evidence-labeled SciFact queries, BM25 reached mean Recall@10 of .930 and lightweight hybrid retrieval reached .864. On a separate 80-claim neural comparison within SciFact development data, a learned passage filter emitted 18 decisive answers. Eleven agreed with the gold verdict and six supplied an aligned citation. We give the aggregation invariant, its assumptions, a retrieval counterexample, and paired claim-level error analysis. The evidence supports a bounded duplication result and identifies citation grounding as a remaining obstacle to reliable scientific verification.

**Comments**

31 pages. Includes a follow-up evaluation of VERITAS v0.2.0. Code and data instructions at https://github.com/Rahulchaube1/veritas.

The archive is prepared for author submission. During the arXiv workflow the submitting author selects the distribution license, confirms authorship and rights, checks any endorsement requirement, and reviews arXiv’s rendered proof before submission. The repository’s MIT software license does not select the manuscript license for the author. Leave DOI, journal reference and related publication fields empty unless applicable. [arXiv submission](https://arxiv.org/submit) · [LaTeX instructions](https://info.arxiv.org/help/submit_tex.html)

To regenerate with XeLaTeX and BibTeX installed, run from the repository root:

```bash
python paper/submission/build.py paper/submission/veritas-arxiv.tar.gz
```
