# Closest comparisons

| Work | Shared question | Difference in the released experiment |
|---|---|---|
| Dong et al., source dependence | How does copying affect aggregation? | We cap a connected component’s contribution per stance and record the passages and metadata behind each edge. Their model estimates dependence and truth iteratively. [Paper](https://www.vldb.org/pvldb/vol2/vldb09-pvldb47.pdf) |
| CONFACT | How should a verifier handle conflicting reports? | We perturb copy count and supplied validity intervals. Their evaluation studies source credibility with conflicting evidence. [Paper](https://www.ijcai.org/proceedings/2025/1073) |
| SAFE and ALCE | Which facts can citations support? | We return exact character offsets and score agreement with annotated SciFact rationales. Long-form and open-web generation require separate experiments. [SAFE](https://arxiv.org/abs/2403.18802) · [ALCE](https://arxiv.org/abs/2305.14627) |
| LiveFact and FreshQA | What can a verifier know at a given time? | Our experiments supply explicit validity intervals. Determining those intervals from independent web archives remains open. [LiveFact](https://aclanthology.org/2026.acl-long.546/) · [FreshLLMs](https://arxiv.org/abs/2310.03214) |
| VeriTaS, ACL 2026 | Fact checking under realistic conditions | That project curates a dynamic multimodal benchmark. This repository is a local text verification engine plus a synthetic experimental fixture. [Proceedings](https://aclanthology.org/2026.acl-long.1948/) |
