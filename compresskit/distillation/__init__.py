"""Knowledge distillation: recover accuracy a pruned or quantized student lost
by training it to match the original (uncompressed) model's outputs, not just
the ground-truth labels.
"""

from .distill import DistillConfig, cosine_lr, dkd_loss, distill, kd_loss

__all__ = [
    "DistillConfig",
    "cosine_lr",
    "dkd_loss",
    "distill",
    "kd_loss",
]
