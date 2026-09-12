"""Human Action Recognition — a real fine-tuned model, not a CIFAR toy.

Architecture is copied verbatim from
`resnet18/resnet-18-on-human-action-recognition-dataset.ipynb` (torchvision
resnet18, ImageNet stem, 224x224, 15-class head) so a state_dict trained there
loads here with no shape mismatch. Unlike `models.py`/`data.py`, which the
toolkit's own CIFAR baselines use, this is a *second* real-world subject to
validate the pipeline generalises past 32x32 CIFAR — the input_shape it needs
(1, 3, 224, 224) is a parameter throughout compresskit, not hardcoded.

Dataset: Kaggle "Human Action Recognition (HAR) Dataset"
(meetnagadia/human-action-recognition-har-dataset), 15 classes.

As downloaded, this is NOT an ImageFolder layout: `train/` is a flat folder of
12,601 images labelled by `Training_set.csv` (filename,label), and `test/` is
a flat folder of 5,410 images with `Testing_set.csv` giving filenames only —
no labels. That test split is a Kaggle submission target, not something we
have ground truth for, so it is useless for measuring accuracy and is not
used anywhere below.

Instead, the 12,601 *labelled* images are split ourselves into a fit portion
and a held-out test portion, deterministically by filename (sort each class's
files, take the last `test_frac` as test) rather than by a seeded RNG — this
must produce the identical split in this file and in
`colab/train_resnet18_har.ipynb`, which duplicates this logic rather than
importing it (Colab can't import this repo), and a hash/sort-based split is
the one thing guaranteed to agree across two separate Python/torch
installations without any shared random state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

HAR_CLASSES = 15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TEST_FRACTION = 0.15

DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data" / "har" / "Human Action Recognition"

_train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

_eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def build_har_resnet18(num_classes: int = HAR_CLASSES, pretrained: bool = True) -> nn.Module:
    """Same architecture as the notebook: torchvision resnet18 + BatchNorm1d/Linear head.

    `pretrained` should be True to construct the architecture for loading a
    trained checkpoint's state_dict, and can be False for a from-scratch build
    (e.g. inside export/inference code where downloading ImageNet weights
    first just to overwrite them is wasted network time).
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def _require_root(root: Path) -> Path:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"HAR dataset not found at {root}. Download it first:\n"
            f"  ./.venv/bin/kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset "
            f"-p data/har --unzip"
        )
    return root


def _read_labels(root: Path) -> dict[str, str]:
    """filename -> label, from Training_set.csv (the only labelled split)."""
    with open(root / "Training_set.csv", newline="") as f:
        return {row["filename"]: row["label"] for row in csv.DictReader(f)}


def class_names(root: Path = DEFAULT_ROOT) -> list[str]:
    labels = _read_labels(_require_root(root))
    return sorted(set(labels.values()))


def _split_filenames(root: Path, test_frac: float = TEST_FRACTION) -> tuple[list, list]:
    """Deterministic per-class split by filename — no RNG, so it reproduces
    identically on Colab and locally. Returns (fit_rows, test_rows), each a
    list of (filename, label)."""
    labels = _read_labels(root)
    by_class: dict[str, list[str]] = {}
    for filename, label in labels.items():
        by_class.setdefault(label, []).append(filename)

    fit_rows, test_rows = [], []
    for label, filenames in by_class.items():
        filenames = sorted(filenames)
        n_test = max(1, round(len(filenames) * test_frac))
        for filename in filenames[:-n_test]:
            fit_rows.append((filename, label))
        for filename in filenames[-n_test:]:
            test_rows.append((filename, label))
    return fit_rows, test_rows


class _HARDataset(Dataset):
    def __init__(self, root: Path, rows: list[tuple[str, str]], class_to_idx: dict[str, int], transform):
        self.image_dir = root / "train"  # both splits come from the labelled train/ folder
        self.rows = rows
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        filename, label = self.rows[index]
        image = Image.open(self.image_dir / filename).convert("RGB")
        return self.transform(image), self.class_to_idx[label]


def _class_to_idx(root: Path) -> dict[str, int]:
    return {name: i for i, name in enumerate(class_names(root))}


def train_loader(batch_size: int = 64, root: Path = DEFAULT_ROOT, num_workers: int = 0) -> DataLoader:
    root = _require_root(root)
    fit_rows, _ = _split_filenames(root)
    ds = _HARDataset(root, fit_rows, _class_to_idx(root), _train_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)


def test_loader(batch_size: int = 64, root: Path = DEFAULT_ROOT, num_workers: int = 0) -> DataLoader:
    root = _require_root(root)
    _, test_rows = _split_filenames(root)
    ds = _HARDataset(root, test_rows, _class_to_idx(root), _eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def calibration_loader(n: int = 512, batch_size: int = 32, root: Path = DEFAULT_ROOT,
                       seed: int = 0) -> DataLoader:
    """Held-out-from-test calibration data, drawn from the fit (train) split.

    Same rule as compresskit/data.py: quantization calibration and BN
    re-estimation both fit parameters to what they see, so this must never be
    the test split, or the reported accuracy stops being honest.
    """
    root = _require_root(root)
    fit_rows, _ = _split_filenames(root)
    ds = _HARDataset(root, fit_rows, _class_to_idx(root), _eval_tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)
