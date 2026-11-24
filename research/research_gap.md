# Testable research question

For the same indexed collection, retrieval budget and stance classifier, how much does the verdict change if one report is syndicated sixteen times? Can an algorithm preserve the balance of observations while retaining recall when publishers legitimately corroborate an event?

We separate three experiments. The first fixes all retrieved evidence and varies only copy count. The second permits ranking and truncation to change the candidate set. The third evaluates document retrieval and citation alignment on independent annotations. The first experiment is a necessary software property. The second measures an end-to-end failure mode. The third measures a different external validity criterion.

The motivating dependence problem and conflict-assessment task already exist in the literature. The unanswered question for this particular implementation is how its clustering errors, temporal exclusions and retrieval cutoffs interact on independently annotated real-world stories. The current controlled benchmark supplies group and interval labels by construction. SciFact supplies sentence rationales, without labels for publisher independence. [Dong et al.](https://www.vldb.org/pvldb/vol2/vldb09-pvldb47.pdf) · [CONFACT](https://www.ijcai.org/proceedings/2025/1073) · [SciFact](https://arxiv.org/abs/2004.14974)
