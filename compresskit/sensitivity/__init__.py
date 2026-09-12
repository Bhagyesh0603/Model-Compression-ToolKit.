"""Per-layer sensitivity analysis: which layers are safe to compress hard and
which are carrying the accuracy, ranked by how much a fake INT8 quantization
of that one layer perturbs the model's output. Drives both quantization
exclusion lists and per-layer pruning ratio overrides.
"""

from .analysis import LayerSensitivity, SensitivityReport, analyze, fake_quantize

__all__ = [
    "LayerSensitivity",
    "SensitivityReport",
    "analyze",
    "fake_quantize",
]
