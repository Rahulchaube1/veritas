# Independent research note for Google Research

**Project:** VERITAS — Verified Evidence Retrieval with Independent Temporal Attribution and Scoring  
**Project founder:** Rahul Chaube  
**Repository:** https://github.com/Rahulchaube1/veritas

Dear Google Research team,

I’m sharing a reproducible experiment on a failure mode in evidence retrieval. When several pages repeat the same report, a verifier can confuse the number of pages with the number of independent observations. Our implementation groups potentially dependent sources, checks temporal eligibility, and keeps the passages supporting each decision available for inspection.

Adding up to sixteen copies of one opposing report left the evidence balance unchanged when all observations were retrieved. Under a restrictive retrieval budget, the copies could still crowd out the primary source. The repository contains both experiments.

Independent scientific annotations exposed a second issue. On 80 SciFact development claims, the passage-filtered neural verifier returned 18 decisive answers, 11 of which matched the gold verdict. Six of its 19 citations matched annotated rationale sentences. These results motivate work on finding independent primary evidence and validating individual citations before presenting a verdict.

The repository includes the implementation, paper, machine-readable results, data provenance and a runnable Colab notebook. I would welcome feedback on how to evaluate the copy-crowding failure on temporally versioned web evidence with independently annotated source relationships.

Rahul Chaube  
Independent VERITAS project
