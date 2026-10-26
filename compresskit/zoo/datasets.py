"""One interface over the datasets the CLI can run: CIFAR-10, HAR, or any image folder.

The CLI used to hardcode CIFAR-10. This lets `compresskit train/pipeline/compress`
run the same flow on a HAR ResNet-18 or on a user's own classification dataset,
without a per-dataset script.

`folder` expects one sub-directory per class (the torchvision ImageFolder
layout). The train/test split is deterministic by sorted filename, with the last
`TEST_FRACTION` of each class held out — no RNG, so the split is identical on
every machine, the same rule `har.py` uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

TEST_FRACTION = 0.15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataBundle:
    name: str
    input_shape: tuple[int, ...]
    num_classes: int
    class_names: list[str]
    train_loader: callable
    test_loader: callable
    calibration_loader: callable


def _folder_split(root: Path) -> tuple[list[int], list[int], ImageFolder]:
    full = ImageFolder(str(root))
    by_class: dict[int, list[tuple[str, int]]] = {}
    for index, (path, label) in enumerate(full.samples):
        by_class.setdefault(label, []).append((path, index))
    fit, held = [], []
    for label in sorted(by_class):
        entries = sorted(by_class[label])
        cut = max(1, int(round(len(entries) * TEST_FRACTION)))
        held += [i for _, i in entries[-cut:]]
        fit += [i for _, i in entries[:-cut]]
    return fit, held, full


class _Transformed(Dataset):
    def __init__(self, base: ImageFolder, indices: list[int], transform):
        self.base, self.indices, self.transform = base, indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        path, label = self.base.samples[self.indices[i]]
        return self.transform(self.base.loader(path)), label


def folder_bundle(root: str | Path, input_size: int = 224, num_workers: int = 2) -> DataBundle:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"image folder dataset not found: {root}")
    fit, held, full = _folder_split(root)
    if not held or not fit:
        raise ValueError(f"{root} needs several images per class to split train/test")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)), transforms.CenterCrop(input_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def train_loader(batch_size=64):
        return DataLoader(_Transformed(full, fit, train_tf), batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)

    def test_loader(batch_size=64):
        return DataLoader(_Transformed(full, held, eval_tf), batch_size=batch_size, num_workers=num_workers)

    def calibration_loader(n=512, batch_size=32, seed=0):
        # Calibration fits parameters to what it sees, so it comes from the
        # train split only — never the held-out test portion.
        ds = _Transformed(full, fit, eval_tf)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(ds), generator=g)[:n].tolist()
        return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)

    return DataBundle("folder", (1, 3, input_size, input_size), len(full.classes), list(full.classes),
                      train_loader, test_loader, calibration_loader)


def cifar10_bundle(root=None) -> DataBundle:
    from . import data

    kw = {"root": root} if root else {}
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return DataBundle(
        "cifar10", (1, 3, 32, 32), 10, names,
        lambda batch_size=128: data.train_loader(batch_size=batch_size, **kw),
        lambda batch_size=256: data.test_loader(batch_size=batch_size, **kw),
        lambda n=512, batch_size=64, seed=0: data.calibration_loader(n=n, batch_size=batch_size, **kw),
    )


def har_bundle(root=None) -> DataBundle:
    from . import har

    kw = {"root": root}
    return DataBundle(
        "har", (1, 3, 224, 224), har.HAR_CLASSES, har.class_names(root),
        lambda batch_size=64: har.train_loader(batch_size=batch_size, **kw),
        lambda batch_size=64: har.test_loader(batch_size=batch_size, **kw),
        lambda n=512, batch_size=32, seed=0: har.calibration_loader(n=n, batch_size=batch_size, seed=seed, **kw),
    )


def get_bundle(name: str, root: str | Path | None = None, input_size: int = 224) -> DataBundle:
    if name == "cifar10":
        return cifar10_bundle(root)
    if name == "har":
        return har_bundle(root)
    if name == "folder":
        if root is None:
            raise ValueError("--dataset folder needs --data-root pointing at one-folder-per-class images")
        return folder_bundle(root, input_size)
    raise ValueError(f"unknown dataset {name!r}; choose cifar10, har or folder")
