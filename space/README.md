---
title: VERITAS Evidence Lab
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
license: mit
short_description: Test how copied ferry reports affect a verification verdict.
---

Change how many copies of a conflicting report enter a search and compare ordinary voting with source grouping. Each result runs VERITAS v0.2.0 on a controlled, fictional register. The public demo has a fixed corpus and uses CPU only.

[Algorithm and evidence](https://github.com/Rahulchaube1/veritas/blob/v0.2.0/docs/verification-v0.2.md) · [Report a reproduction](https://github.com/Rahulchaube1/veritas/issues/1)

To rebuild, copy the v0.2.0 veritas_evidence wheel from the [official release](https://github.com/Rahulchaube1/veritas/releases/tag/v0.2.0) beside this Dockerfile, then run `docker build -t veritas-space .` and `docker run --rm -p 7860:7860 veritas-space`.

The GitHub workflow **Publish Hugging Face Space** uses the repository secret HF_TOKEN to create the free public Docker Space at oncody/veritas-evidence-lab or update the existing demo. The account needs permission to create and write the target repository.
