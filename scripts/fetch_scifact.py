"""Materialize only the named JSONL members of AI2's official SciFact release."""

import hashlib
import io
import tarfile
from pathlib import Path
from urllib.request import urlopen

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SHA = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"


def fetch(target: Path):
    with urlopen(URL, timeout=60) as handle:
        data = handle.read(8_000_001)
    if hashlib.sha256(data).hexdigest() != SHA:
        raise ValueError("Unexpected archive digest; review upstream release before updating")
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        for name in ["corpus", "claims_train", "claims_dev"]:
            member = archive.extractfile(f"data/{name}.jsonl")
            if member is None:
                raise ValueError(name)
            (target / (name + ".jsonl")).write_bytes(member.read())


if __name__ == "__main__":
    fetch(Path(".cache/scifact"))
