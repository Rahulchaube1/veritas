# From a query to a verdict

1. Validate input, metadata and requested budget. Distinguish atomic assertions, a limited yes/no grammar and single-value copular questions. Interpret explicit ISO as-of dates. Unsupported question forms trigger abstention.
2. Retrieve an increasingly large prefix of the fused ranking until the policy stops. Record the query actions. Upserts transactionally replace SQLite rows and invalidate the in-process index.
3. Split source text into passages with original character offsets. Select up to five per document using query-token Jaccard similarity. Exact statements, opposing negation and unambiguous single numeric slots have deterministic rules. Production inference can add a pretrained NLI cross-encoder. The optional scientific rationale filter precedes neural classification.
4. Construct candidate graph edges through an inverted index over metadata and token trigrams. Near copies receive a similarity-based suspicion label. Strong links cap votes. A shared publisher also caps a vote, which can suppress legitimately independent reporting.
5. Aggregate eligible passages by connected component and stance. Report conflicting eligible passages, signed balance, the retrieval budget and exact citations. Unresolved provenance and unknown factual confidence remain visible.
6. Check each cited span against the current document snapshot before emitting an extractive claim verdict. A valid character span can still be semantically irrelevant. External rationale alignment measures this failure.

The output uses zero-based positions in the evidence array for citations and contradictions. A graph edge points to a comparison document. Chronology orders comparisons where dates exist. Group membership expresses a conservative aggregation decision rather than proof of authorship or copying. [Implementation](../src/veritas/core.py)

The verify endpoint also splits sentence-separated assertions and shares a single retrieval budget across them. Deep search can follow locally indexed citation IDs and expand a query using terms from the leading document title. Composite reasoning from unrelated sentences remains an open evaluation target.
