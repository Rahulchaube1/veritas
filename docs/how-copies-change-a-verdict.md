# When eight search results come from one report

**Rahul Chaube** · Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, India

Imagine checking a ferry’s passenger capacity. A register says 40. Eight web pages say 41, using the same wording. If all eight repeat one observation, you have two accounts to reconcile. Whether you ever see the register also depends on which pages the search engine puts in front of you.

VERITAS lets you change those two parts of the problem separately. You can keep the retrieved documents fixed and change how they vote. Or you can keep the voting rule fixed and change which documents get inspected. Both experiments use the same release code. [Run the notebook](https://colab.research.google.com/github/Rahulchaube1/veritas/blob/v0.2.0/notebooks/veritas.ipynb)

## Counting one observation once

The verifier first finds passages that support or contradict the claim and checks temporal eligibility. It groups reports using observable signals such as identical text or an explicit citation. Each group contributes its strongest eligible supporting passage and its strongest eligible contradicting passage. If their sums across groups are S and C, the signed evidence balance is (S − C)/(1 + S + C).

For the claim “The ferry carries 40 passengers,” ordinary page voting gives S = 1 and C = 8. The balance is −7/10. Group the eight copies together and the sums are 1 and 1. The balance is zero, and the verdict is conflict. The useful thing to show the reader at that point is the register beside the opposing passage. Neither the formula nor the vote determines the actual capacity of the ferry.

There is a precise guarantee here. If a new copy enters an existing group, changes no other group or eligibility decision, and cannot raise either maximum, both sums stay fixed. That guarantee depends on the evidence already being visible. [Formal argument](../research/mathematical_foundations.md)

## What if the register never reaches the verifier?

Give search one step and enough space for two documents. If the top two are copies, grouping can reduce them to one observation, but the verifier still sees evidence on only one side. The v0.2 retrieval option looks farther down the ranking and gives previously unseen full texts priority before allocating the same two inspection slots.

We varied copy count across five settings and the search budget across three. Ordinary ranking exposed the disagreement in 7 of the 15 combinations. Prioritizing distinct texts exposed it in all 15. The experiment uses controlled copies. A paraphrase may escape the exact-text check, and an independent report outside the candidate pool can still be missed. [Experiment settings and outcomes](verification-v0.2.md)

## An accuracy gain can carry a cost

In a separate experiment, two scientific verification profiles received the same retrieved documents on 172 public SciFact claims. The revised profile scored individual sentences and changed its inference model. It produced 18 correct answers with at least one citation matching an annotated rationale, up from 9. It also produced 19 incorrect decisive answers, up from 11. Correctness among decisive answers fell from 22/33 to 34/53. The changes in this experiment came from the verification profile. They cannot be attributed to the ferry experiment’s duplicate-selection rule. [Paired results](../results/scientific_172_summary.json)

Each recorded citation has a document ID and character offsets. A follow-up implementation check found that comparing passage text alone could credit the wrong occurrence if the same words appeared twice in an abstract. We now check that the cited offsets fall inside the annotated sentence. Replaying all 99 citations against the official corpus preserved the measured totals of 9/37 and 18/62. The added regression test exercises a duplicated sentence directly. [Audit implementation](../experiments/rationale_spans.py) · [Regression test](../tests/test_rationale_spans.py)

## A question for independent reviewers

Can a verifier tell when two outlets copied a common source and when two observers independently reported the same value? Exact matching alone cannot resolve that distinction. A public collection with human judgments about source dependence would let us measure both kinds of error. If you know such a collection, or if a clean CPU run gives different counts, the reproduction thread records the inputs and the expected outputs. [Independent reproduction invitation](https://github.com/Rahulchaube1/veritas/issues/1)
