# Mathematical specification

For query q and documents D, let BM25 use k₁ = 1.2, b = 0.75 and IDF(t) = log(1 + (N − df(t) + 0.5)/(df(t) + 0.5)). Distinct query terms contribute once. Character TF-IDF followed by a rank-at-most-64 truncated SVD provides the offline second ranking. For small collections it uses the TF-IDF vectors directly. A pinned multilingual sentence encoder with masked mean pooling, L2 normalization and HNSW maximum-inner-product search provides the neural ranking.

The fusion score is Σⱼ 1/(60 + rankⱼ(d)), omitting lists where d is absent. Optional additions are p(relevant | features)/60 and 0.05 × sigmoid(cross-encoder logit). Only the relevance coefficients are fitted by the project. The fusion multipliers are design choices. Logistic models minimize regularized binary log loss. The scientific rationale filter selects its threshold by validation F₀.₅ on passage annotations.

Let t be an explicit date or the evaluation date. A document is ineligible if its publication or last modification is later than t, or t lies outside its supplied inclusive validity bounds. A temporally scoped claim also requires at least one supplied bound. Event dates are retained as metadata. They alone impose no validity interval.

Construct components from conservative dependency links between examined documents. Let uᵢ and vᵢ be the respective maximum eligible support and contradiction strengths in component i, using zero for an empty set. S = Σuᵢ, C = Σvᵢ and m = (S − C)/(1 + S + C). The extra one shrinks the balance towards zero when evidence is sparse. Its magnitude is an evidence statistic. It has no probabilistic interpretation by itself.

Return insufficient when both totals vanish. Return conflict when both are positive and min(S,C)/max(S,C) ≥ 0.34. Otherwise return the sign of S−C, subject to a surviving citation. The 0.34 cutoff is hand-set. For the controlled population, fit a nondecreasing function g of |m| by least-squares isotonic regression on calibration cases with a decisive answer and known simulator truth. The target is whether that answer agrees with simulator truth. Coverage is reported over all test cases.

**Conditional invariance.** Hold retrieved documents, temporal eligibility, component assignments and all existing stance strengths fixed. Inserting a duplicate into its original component with a strength no larger than the existing maximum for its stance leaves both maxima, S, C and m unchanged. This follows from max(x,x)=x. Retrieval truncation, component bridges and model rescoring invalidate these assumptions.

The finite search budget ensures termination. The default heuristic stops after exhausting candidates or, after at least two rounds, reaching a decisive margin of at least 0.65 in absolute value. This approximates a decision that further retrieval is unlikely to pay off. It supplies neither an optimal utility policy nor a convergence theorem.
