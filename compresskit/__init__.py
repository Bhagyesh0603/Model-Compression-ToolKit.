"""compresskit — a CPU-latency-aware CNN compression toolkit.

Pipeline: prune -> BN recover -> distill -> quantize, with every stage exported
to ONNX and measured on real hardware rather than estimated from FLOPs.
Quantization is always last, because it changes weight dtype and blocks further
pruning and fine-tuning.

    import compresskit as ck

    ck.probe()                                  # what this CPU can do
    ck.compress(model, calib, test,             # budget-driven, stops early
                budget="latency<1ms,accuracy_drop<1%")
    ck.Pipeline(model, calib, test).prune(0.5).recover().quantize().run()
"""

from .benchmark import Latency, Measurement, benchmark, compression_score, efficiency_score, measure
from .distill import DistillConfig, dkd_loss, distill, kd_loss
from .export import count_parameters, to_onnx
from .host import HostContext, probe
from .pipeline import Pipeline, PipelineResult
from .prune import PruneConfig, StructuredPruner, prune
from .quantize import QuantConfig, quantize
from .recover import recalibrate_bn
from .search import Budget, SearchResult, compress
from .sensitivity import SensitivityReport, analyze

__version__ = "0.1.0"

__all__ = [
    "Budget",
    "DistillConfig",
    "HostContext",
    "Latency",
    "Measurement",
    "Pipeline",
    "PipelineResult",
    "PruneConfig",
    "QuantConfig",
    "SearchResult",
    "SensitivityReport",
    "StructuredPruner",
    "analyze",
    "benchmark",
    "compress",
    "compression_score",
    "count_parameters",
    "distill",
    "dkd_loss",
    "efficiency_score",
    "kd_loss",
    "measure",
    "probe",
    "prune",
    "quantize",
    "recalibrate_bn",
    "to_onnx",
]
