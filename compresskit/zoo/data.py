from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# CIFAR-10 is small enough to live in RAM and its transforms are cheap, so worker
# processes cost more in spawn/IPC overhead than they save: measured on this host,
# num_workers=0 runs an epoch in 14.2s vs 24.0s with two workers, and avoids the
# worker-abort crashes that spawn-based loading hits on macOS.
def train_loader(batch_size=128, root=DEFAULT_ROOT, num_workers=0):
    ds = datasets.CIFAR10(str(root), train=True, download=True, transform=_train_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, drop_last=True)


def test_loader(batch_size=256, root=DEFAULT_ROOT, num_workers=0):
    ds = datasets.CIFAR10(str(root), train=False, download=True, transform=_eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def calibration_loader(n=512, batch_size=64, root=DEFAULT_ROOT, seed=0):
    """Held-out-from-test calibration data, drawn from the TRAIN split.

    Quantization calibration and BN re-estimation must never touch the test set,
    or the reported accuracy stops being an honest generalization estimate.
    """
    ds = datasets.CIFAR10(str(root), train=True, download=True, transform=_eval_tf)
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n].tolist()
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False, num_workers=0)
