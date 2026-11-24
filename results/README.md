# Reading the measurements

* **scifact.json** uses independent query-level means, with nonparametric bootstrap intervals over 188 eligible SciFact queries. The retriever internally named dense in this file uses corpus-fitted character LSA. The learned ranker is incremental to the rank-fusion score.
* **neural_scifact.json** records a 48-claim diagnostic. **neural_gated.json** evaluates the rationale gate on 80 other claims and reruns the ungated verifier on the same 80. Correct overall verdicts include correct insufficient-evidence decisions. Answered counts only decisive support or refutation.
* **extractive_baselines.json** evaluates four-document sentence extraction. It measures alignment to annotated rationales, including claims with no annotated rationale. The final two baselines abstained on every query. The separate-search method spent one additional call per claim.
* **controlled.json** reports synthetic verdict labels and a separate accuracy target from latent simulator truth. The mean of one retrieval round includes queries that used zero steps and others that used two. With known truth, the system made 150 decisive test predictions. The synthetic calibrator's low error cannot be transferred to a real corpus.
* **candidate_crowding.json** isolates top-K exclusion. A primary source can disappear from view while grouped voting continues to respect the within-visible-set duplicate cap. This is an end-to-end failure.
* **ranker_training.json** and **rationale_training.json** report fits on disjoint source groups. **validation.json** records the local release checks. Direct software dependencies are listed in configs/constraints.txt.

All experiment files report measurements. Re-running timing experiments changes the recorded durations.
