"""Fetch pinned optional ONNX checkpoints. Keep third-party weights outside Git."""

import argparse
import shutil
from pathlib import Path
from urllib.request import urlopen

MODELS = {
    "dense": ("paraphrase-multilingual-MiniLM-L12-v2", "2c4055b12046f11709e9df2c122e59ffbdc2f900"),
    "nli": ("nli-deberta-v3-xsmall", "2a4f614a701367a02d51389039afc998faeda637"),
    "rerank": ("ms-marco-MiniLM-L-6-v2", "a09144355adeed5f58c8ed011d209bf8ee5a1fec"),
}


def fetch(name, root):
    model, revision = MODELS[name]
    for file in ["config.json", "tokenizer.json", "onnx/model_quantized.onnx"]:
        path = root / name / file
        if path.is_file():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/Xenova/{model}/resolve/{revision}/{file}"
        with urlopen(url, timeout=90) as response, path.with_suffix(".part").open("wb") as out:
            shutil.copyfileobj(response, out)
        path.with_suffix(".part").replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=MODELS)
    parser.add_argument("--root", type=Path, default=Path(".cache/models"))
    args = parser.parse_args()
    for name in args.names:
        fetch(name, args.root)
