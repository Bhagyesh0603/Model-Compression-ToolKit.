"""Post-training INT8/FP16 quantization through ONNX Runtime — always the last
compression stage, since it changes weight dtype and blocks further pruning
or gradient-based fine-tuning.
"""

from .quantize import QuantConfig, QuantResult, quantize

__all__ = [
    "QuantConfig",
    "QuantResult",
    "quantize",
]
