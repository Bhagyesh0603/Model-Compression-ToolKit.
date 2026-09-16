"""zoo/ path resolution: DEFAULT_ROOT must land on the repo's real data/ dir.

`git mv compresskit/data.py compresskit/zoo/data.py` (and the har.py
equivalent) moved these files one directory deeper without updating their
`.parent.parent`-based relative path to the repo root, so DEFAULT_ROOT
silently resolved to a nonexistent `compresskit/data` instead of the real,
already-downloaded `data/` at the repo root.
"""

from pathlib import Path

from compresskit.zoo import data, har

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cifar_default_root_resolves_to_the_repo_root_data_dir():
    assert data.DEFAULT_ROOT == REPO_ROOT / "data"


def test_har_default_root_resolves_to_the_repo_root_data_dir():
    assert har.DEFAULT_ROOT == REPO_ROOT / "data" / "har" / "Human Action Recognition"
