# Changes

## 0.2.0 · 24 November 2026

Adds typed English–Spanish count matching, two-document count verification with both source spans, and optional duplicate-aware candidate selection. Source questions cannot vote as assertions. An optional scientific profile scores individual passages and applies a FEVER-trained NLI model. The expanded 31-page paper includes a follow-up experiment. The versioned Colab notebook reruns the 360 synthetic cases and displays the paired 172-claim SciFact comparison. [Measurements and limitations](docs/verification-v0.2.md)

## 0.1.0 · 24 November 2026

Initial independent research release. Includes local retrieval, evidence grouping, temporal eligibility, optional ONNX inference, a trained rationale filter, calibrated synthetic experiments, SciFact evaluation, API, CLI, web interface and reproducibility materials.
