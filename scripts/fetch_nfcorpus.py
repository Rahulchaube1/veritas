"""Fetch the BEIR maintainers' pinned NFCorpus mirror and official test judgments."""

import json
from pathlib import Path
from urllib.request import urlopen

import pyarrow.parquet as pq

ROOT = Path(".cache/nfcorpus")
BASE = "https://huggingface.co/datasets/BeIR/"


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    for kind in ("corpus", "queries"):
        name = f"{kind}/{kind}-00000-of-00001.parquet"
        url = BASE + "nfcorpus/resolve/b5026a0e96e8a7ac4f95f482a596389289d46269/" + name
        parquet = ROOT / (kind + ".parquet")
        with urlopen(url, timeout=45) as source:
            parquet.write_bytes(source.read())
        with (ROOT / (kind + ".jsonl")).open("w") as output:
            for batch in pq.ParquetFile(parquet).iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    output.write(json.dumps(row) + "\n")
    path = ROOT / "qrels" / "test.tsv"
    path.parent.mkdir(exist_ok=True)
    with urlopen(
        BASE + "nfcorpus-qrels/resolve/a451b3b26d3ae1358f259c1a3a4dd61fcea35a65/test.tsv", timeout=30
    ) as r:
        path.write_bytes(r.read())


if __name__ == "__main__":
    main()
