# Expanded VERITAS technical report

[Read the PDF](veritas.pdf). The report studies software v0.1.0 and adds a paired replay of the two neural configurations on the original 80 evaluation IDs. The expanded report and replay are published on the repository’s main branch.

## Build the PDF

With XeLaTeX and BibTeX installed, run the following in this directory. The tracked PDF figures are sufficient for an ordinary build.

```bash
xelatex -halt-on-error veritas.tex
bibtex veritas
xelatex -halt-on-error veritas.tex
xelatex -halt-on-error veritas.tex
```

The main file sets the title, author, affiliation, typography and contents. The included sources contain the [method](method.tex), [experiments](experiments.tex), [interpretation](interpretation.tex) and [reference procedure](appendix.tex). References are drawn from the [project bibliography](../research/references.bib).

## Replay the paired analysis

From the repository root, follow the [neural reproduction setup](../docs/reproduce.md) to install the project and fetch the official SciFact archive and the pinned ONNX models. The two commands below refresh the [paired query records](paired_audit.json) and vector figures. The second command requires Matplotlib.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=17 python paper/paired_audit.py
MPLBACKEND=Agg python paper/figures.py
```

The audit uses the 80 archived IDs in their original order. Each record includes the gold verdict, both predicted verdicts, citation counts, counts aligned to a gold rationale, and whether the gated result reproduced its published record. The two-sided exact paired test conditions on queries where correctness differs between configurations. This is a replay on public development data. The original measured runtimes are preserved in the [release results](../results/).
