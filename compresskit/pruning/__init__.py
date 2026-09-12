"""Structured channel pruning: remove whole channels via Torch-Pruning's
dependency graph, then recalibrate BatchNorm statistics to recover the free
accuracy the removed channels' stale running stats would otherwise cost.
"""

from .prune import CRITERIA, PruneConfig, PruneStats, StructuredPruner, classifier_layers, prune
from .recover import recalibrate_bn

__all__ = [
    "CRITERIA",
    "PruneConfig",
    "PruneStats",
    "StructuredPruner",
    "classifier_layers",
    "prune",
    "recalibrate_bn",
]
