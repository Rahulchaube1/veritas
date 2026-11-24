"""Publish the fixed VERITAS CPU demo using a repository-write credential."""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

SPACE = "oncody/veritas-evidence-lab"


def publish(folder: Path):
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set the repository secret HF_TOKEN with permission to create and write this Space.")
    expected_files = {
        "README.md",
        "Dockerfile",
        "app.py",
        "index.html",
        "veritas_evidence-0.2.0-py3-none-any.whl",
    }
    if {p.name for p in folder.iterdir()} != expected_files:
        raise SystemExit(
            "Unexpected deployment contents. Use a directory with the Space files and the release wheel."
        )
    api = HfApi(token=token)
    if api.whoami()["name"] != SPACE.split("/")[0]:
        raise SystemExit("The credential does not belong to the configured Space owner.")
    if not api.repo_exists(SPACE, repo_type="space"):
        api.create_repo(SPACE, repo_type="space", space_sdk="docker", private=False)
    else:
        info = api.repo_info(SPACE, repo_type="space")
        existing = {s.rfilename for s in info.siblings}
        if existing - {".gitattributes", "README.md"}:
            card = info.card_data
            if card is None or card.get("title") != "VERITAS Evidence Lab":
                raise SystemExit("An existing Space with different content occupies this name.")
    change = api.upload_folder(
        folder_path=folder,
        repo_id=SPACE,
        repo_type="space",
        commit_message="Publish the VERITAS evidence lab",
    )
    print(change.commit_url)


if __name__ == "__main__":
    publish(Path(sys.argv[1]))
