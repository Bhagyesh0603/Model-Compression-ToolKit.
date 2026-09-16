"""Structured channel pruning: remove whole channels via Torch-Pruning's
dependency graph, then recalibrate BatchNorm statistics to recover the free
accuracy the removed channels' stale running stats would otherwise cost.
"""

from .latency_guide import LatencyGuidedReport, LayerLatencyImpact, measure_latency_impact
from .prune import CRITERIA, PruneConfig, PruneStats, StructuredPruner, classifier_layers, prune
from .recover import recalibrate_bn

__all__ = [
    "CRITERIA",
    "LatencyGuidedReport",
    "LayerLatencyImpact",
    "PruneConfig",
    "PruneStats",
    "StructuredPruner",
    "classifier_layers",
    "measure_latency_impact",
    "prune",
    "recalibrate_bn",
]
