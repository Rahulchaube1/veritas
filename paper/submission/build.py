"""Prepare a portable arXiv upload from the checked-out paper and its bibliography."""

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
FIGURES = ["duplicate-injection.pdf", "retrieval-intervals.pdf", "paired-grounding.pdf"]
SECTIONS = ["method.tex", "experiments.tex", "interpretation.tex", "appendix.tex", "followup.tex"]


def build(archive: Path):
    archive = archive.resolve()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        main = (PAPER / "veritas.tex").read_text()
        (stage / "veritas.tex").write_text(main.replace("../research/references", "references"))
        shutil.copyfile(PAPER.parent / "research/references.bib", stage / "references.bib")
        for filename in SECTIONS + FIGURES:
            shutil.copyfile(PAPER / filename, stage / filename)
        (stage / "00README.json").write_text(
            json.dumps(
                {
                    "process": {"compiler": "xelatex"},
                    "sources": [{"filename": "veritas.tex", "usage": "toplevel"}],
                },
                indent=2,
            )
            + "\n"
        )
        latex = ["xelatex", "-interaction=nonstopmode", "-halt-on-error", "veritas.tex"]
        for argv in [latex, ["bibtex", "veritas"], latex, latex]:
            subprocess.run(argv, cwd=stage, check=True, stdout=subprocess.DEVNULL)
        log = (stage / "veritas.log").read_text()
        if "undefined references" in log or "undefined citations" in log:
            raise ValueError("PDF contains unresolved references")
        files = ["00README.json", "veritas.tex", "references.bib", "veritas.bbl", *SECTIONS, *FIGURES]
        with archive.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                for name in sorted(files):
                    info = tarfile.TarInfo(name)
                    info.size = (stage / name).stat().st_size
                    with (stage / name).open("rb") as f:
                        tar.addfile(info, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    build(parser.parse_args().archive)
