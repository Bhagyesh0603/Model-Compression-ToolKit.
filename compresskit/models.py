"""Backward-compat shim for checkpoints saved before the zoo/ package split.

`zoo/train.py`'s `save_checkpoint` pickles the whole `nn.Module`, not just its
state_dict, so the pickle bakes in the class's module path at save time. The
"Phase 6" reorg moved these classes from `compresskit.models` to
`compresskit.zoo.models`, which breaks `torch.load` on every checkpoint saved
before that move with `ModuleNotFoundError: No module named 'compresskit.models'`.
Re-exporting the same names here keeps those old checkpoints loadable.
"""

from .zoo.models import BasicBlock, CifarResNet, CifarVGG, ResNet18Cifar, build  # noqa: F401
