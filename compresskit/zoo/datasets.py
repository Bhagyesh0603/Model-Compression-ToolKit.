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


def _train_tf(size: int):
    return transforms.Compose([
        transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_tf(size: int):
    return transforms.Compose([
        transforms.Resize(int(size * 256 / 224)), transforms.CenterCrop(size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@dataclass
class DataBundle:
    name: str
    input_shape: tuple[int, ...]
    num_classes: int
    class_names: list[str]
    train_loader: callable
    test_loader: callable
    calibration_loader: callable
    val_loader: callable | None = None  # held out of the training split; set by --holdout-val


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

    train_tf, eval_tf = _train_tf(input_size), _eval_tf(input_size)

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


IMAGENET_CLASSES = 1000
DEFAULT_TEST_SAMPLES = 2000


def _strided(total: int, n: int) -> list[int]:
    """`n` evenly spaced indices into `total` items (all of them if n is 0 or >= total)."""
    if 0 < n < total:
        return [int(i * total / n) for i in range(n)]
    return list(range(total))


def imagenet_bundle(root: str | Path, input_size: int = 224, test_samples: int | None = None,
                    num_workers: int = 4) -> DataBundle:
    """ImageNet-1k layout: `root/train/<wnid>/*` and `root/val/<wnid>/*`.

    Built for torchvision's pretrained classifiers, whose output index i is the
    i-th class in *sorted* wnid order. ImageFolder numbers classes the same way,
    so the labels line up only when all 1000 class folders are present — with
    fewer, every label shifts and accuracy reads near 0% without any error. That
    case is refused instead of scored.

    Scoring the 50,000 validation images on CPU once per stage takes hours, so
    the test split is a deterministic evenly-strided subset of `test_samples`
    images (default 2000, 0 = all). At 2000 images the standard error on a
    ~90% accuracy is about 0.7 pp: fine for before/after, not for a 0.2 pp claim.
    """
    root = Path(root)
    train_dir, val_dir = root / "train", root / "val"
    for d in (train_dir, val_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"ImageNet layout needs {d} (one sub-directory per class)")

    train_set = ImageFolder(str(train_dir), _train_tf(input_size))
    calib_set = ImageFolder(str(train_dir), _eval_tf(input_size))
    val_set = ImageFolder(str(val_dir), _eval_tf(input_size))
    if train_set.classes != val_set.classes:
        raise ValueError("train/ and val/ do not contain the same class folders")
    if len(train_set.classes) != IMAGENET_CLASSES:
        raise ValueError(
            f"{len(train_set.classes)} class folders found, expected {IMAGENET_CLASSES}: "
            "pretrained ImageNet weights map output index i to the i-th sorted class, so a "
            "subset of classes would be scored against the wrong labels")

    stride_idx = _strided(len(val_set), DEFAULT_TEST_SAMPLES if test_samples is None else test_samples)

    def train_loader(batch_size=64):
        return DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)

    def test_loader(batch_size=64):
        return DataLoader(Subset(val_set, stride_idx), batch_size=batch_size, num_workers=num_workers)

    def calibration_loader(n=512, batch_size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(calib_set), generator=g)[:n].tolist()
        return DataLoader(Subset(calib_set, idx), batch_size=batch_size, shuffle=False)

    return DataBundle("imagenet", (1, 3, input_size, input_size), IMAGENET_CLASSES, list(train_set.classes),
                      train_loader, test_loader, calibration_loader)


def cifar10_upscaled_bundle(root=None, input_size: int = 224, test_samples: int | None = None,
                            num_workers: int = 2) -> DataBundle:
    """CIFAR-10 resized to `input_size` with ImageNet normalisation.

    For ImageNet-pretrained models (`--arch tv:...`) that need a 224x224 input but
    whose task is CIFAR-10 — the way to run a large model end to end on a Kaggle
    notebook that only has CIFAR-10 attached. 224 is the size VGG's fixed 7x7
    pooling expects; other sizes can fail ONNX export. Upscaling adds no detail,
    it only makes the network large. The test split is a deterministic strided
    subset (default 2000 images, 0 = all 10,000) because scoring the full set on
    CPU at every stage of a 138M-parameter model takes hours.
    """
    from torchvision.datasets import CIFAR10

    from .data import resolve_root

    base = str(resolve_root(root))
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    train_tf = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                   transforms.Resize(input_size), *norm])
    eval_tf = transforms.Compose([transforms.Resize(input_size), *norm])
    train_set = CIFAR10(base, train=True, download=True, transform=train_tf)
    calib_set = CIFAR10(base, train=True, download=True, transform=eval_tf)
    test_set = CIFAR10(base, train=False, download=True, transform=eval_tf)
    test_idx = _strided(len(test_set), DEFAULT_TEST_SAMPLES if test_samples is None else test_samples)

    def train_loader(batch_size=64):
        return DataLoader(train_set, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)

    def test_loader(batch_size=64):
        return DataLoader(Subset(test_set, test_idx), batch_size=batch_size, num_workers=num_workers)

    def calibration_loader(n=512, batch_size=32, seed=0):
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(calib_set), generator=g)[:n].tolist()
        return DataLoader(Subset(calib_set, idx), batch_size=batch_size, shuffle=False)

    return DataBundle("cifar10-upscaled", (1, 3, input_size, input_size), 10, list(train_set.classes),
                      train_loader, test_loader, calibration_loader)


def cifar10_bundle(root=None, holdout_val: int = 0) -> DataBundle:
    from . import data

    kw = {"root": root} if root else {}
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return DataBundle(
        "cifar10", (1, 3, 32, 32), 10, names,
        lambda batch_size=128: data.train_loader(batch_size=batch_size, holdout=holdout_val, **kw),
        lambda batch_size=256: data.test_loader(batch_size=batch_size, **kw),
        lambda n=512, batch_size=64, seed=0: data.calibration_loader(
            n=n, batch_size=batch_size, seed=seed, holdout=holdout_val, **kw),
        (lambda batch_size=256: data.val_loader(holdout_val, batch_size=batch_size, **kw)) if holdout_val else None,
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


def get_bundle(name: str, root: str | Path | None = None, input_size: int = 224,
               test_samples: int | None = None, holdout_val: int = 0) -> DataBundle:
    if holdout_val and name != "cifar10":
        raise ValueError("--holdout-val is implemented for --dataset cifar10 only")
    if name == "cifar10":
        return cifar10_bundle(root, holdout_val)
    if name == "har":
        return har_bundle(root)
    if name == "folder":
        if root is None:
            raise ValueError("--dataset folder needs --data-root pointing at one-folder-per-class images")
        return folder_bundle(root, input_size)
    if name == "cifar10-upscaled":
        return cifar10_upscaled_bundle(root, input_size, test_samples)
    if name == "imagenet":
        if root is None:
            raise ValueError("--dataset imagenet needs --data-root containing train/ and val/")
        return imagenet_bundle(root, input_size, test_samples)
    raise ValueError(f"unknown dataset {name!r}; choose cifar10, cifar10-upscaled, har, folder or imagenet")
