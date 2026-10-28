"""Any torchvision ImageNet classifier as a `--arch tv:<name>` (e.g. tv:vgg16_bn).

This is how the toolkit reaches models far larger than the CIFAR zoo, such as
VGG-16-BN (138M parameters, 528 MB) or ConvNeXt-Large (198M, 790 MB). The
pruning engine itself is architecture-generic (Torch-Pruning's dependency
graph); only model construction and the dataset were CIFAR-specific.

Pretrained weights are torchvision's `DEFAULT` set and are downloaded on first
use, so this needs network access. Depthwise convolutions (ConvNeXt, MobileNet)
prune and run, but their accuracy after pruning has not been measured here.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

TV_PREFIX = "tv:"
IMAGENET_CLASSES = 1000


def is_torchvision_arch(arch: str) -> bool:
    return arch.startswith(TV_PREFIX)


def build_torchvision(arch: str, pretrained: bool = True) -> nn.Module:
    name = arch[len(TV_PREFIX):] if is_torchvision_arch(arch) else arch
    if name not in models.list_models(module=models):
        raise ValueError(f"{name!r} is not a torchvision classification model; "
                         f"see torchvision.models.list_models()")
    return models.get_model(name, weights="DEFAULT" if pretrained else None).eval()


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the final Linear (VGG classifier[6], ResNet fc, ConvNeXt classifier[2]) for a fresh
    `num_classes` head, so a pretrained backbone can be fine-tuned on another task."""
    last_name, last = None, None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name, last = name, module
    if last is None:
        raise ValueError("model has no Linear layer to retarget")
    parent = model
    *path, leaf = last_name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, leaf, nn.Linear(last.in_features, num_classes))
    return model
