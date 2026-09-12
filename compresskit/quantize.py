"""Post-training quantization through ONNX Runtime — always the last stage.

Quantization changes weight dtype, which breaks structural pruning and blocks
further gradient-based fine-tuning, so nothing may run after it.

The precision is chosen from the host probe rather than assumed. On a CPU with
fused INT8 kernels (VNNI/AMX/DotProd/I8MM) INT8 is both smaller and faster; on a
CPU without them it is smaller and often *slower*, so this defaults to FP16 there
and makes INT8 an explicit opt-in with a warning attached to the result.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from .core import host as host_mod
from .core.export import size_mb

CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class LoaderCalibrationReader(CalibrationDataReader):
    """Adapts a torch DataLoader to ORT's calibration interface.

    Must be fed calibration data drawn from the *train* split. Calibrating on
    test data fits activation ranges to the evaluation set and turns the reported
    accuracy into a number that cannot be trusted.
    """

    def __init__(self, loader, input_name: str = "input", batches: int | None = None):
        self.loader = loader
        self.input_name = input_name
        self.batches = batches
        self._iter = None
        self._served = 0
        self.rewind()

    def rewind(self):
        self._iter = iter(self.loader)
        self._served = 0

    def get_next(self):
        if self.batches is not None and self._served >= self.batches:
            return None
        batch = next(self._iter, None)
        if batch is None:
            return None
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        self._served += 1
        return {self.input_name: images.numpy().astype(np.float32)}


@dataclass
class QuantConfig:
    precision: str = "auto"
    per_channel: bool = True
    calibrate: str = "minmax"
    calibration_batches: int | None = None
    reduce_range: bool | None = None
    exclude_nodes: list[str] = field(default_factory=list)
    allow_slow_int8: bool = False

    def __post_init__(self):
        if self.precision not in ("auto", "int8", "fp16", "fp32"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if self.calibrate not in CALIBRATION_METHODS:
            raise ValueError(f"unknown calibration method {self.calibrate!r}; choose from {sorted(CALIBRATION_METHODS)}")


@dataclass
class QuantResult:
    path: Path
    precision: str
    size_mb: float
    quantized_ops: dict[str, int]
    unquantized_ops: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of compute-heavy ops that actually got a low-precision path."""
        total = sum(self.quantized_ops.values()) + sum(self.unquantized_ops.values())
        return sum(self.quantized_ops.values()) / total if total else 0.0

    def summary(self) -> str:
        lines = [f"{self.precision} artifact: {self.path.name} ({self.size_mb:.2f} MB)"]
        if self.precision == "int8":
            lines.append(
                f"  quantized {sum(self.quantized_ops.values())}/"
                f"{sum(self.quantized_ops.values()) + sum(self.unquantized_ops.values())} "
                f"compute ops ({self.coverage * 100:.0f}%)"
            )
            if self.unquantized_ops:
                left = ", ".join(f"{k}x{v}" for k, v in sorted(self.unquantized_ops.items()))
                lines.append(f"  still float: {left}")
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)


# Ops whose cost dominates CNN inference; coverage is reported over these.
_COMPUTE_OPS = {"Conv", "ConvTranspose", "Gemm", "MatMul"}


def _op_coverage(path: Path) -> tuple[dict[str, int], dict[str, int]]:
    """Which compute ops ended up integer vs left in float.

    An op with no fused INT8 kernel is silently left as float by the quantizer.
    The file still shrinks, so a size-only report looks like a success while the
    latency win quietly fails to appear — this is what surfaces that.
    """
    model = onnx.load(str(path))
    quantized: dict[str, int] = {}
    unquantized: dict[str, int] = {}

    nodes = list(model.graph.node)
    producers = {output: node for node in nodes for output in node.output}
    consumers: dict[str, list] = {}
    for node in nodes:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)

    for node in nodes:
        base = node.op_type.replace("Integer", "").replace("QLinear", "")
        if base not in _COMPUTE_OPS:
            continue
        if node.op_type.startswith("QLinear") or node.op_type.endswith("Integer"):
            quantized[base] = quantized.get(base, 0) + 1
            continue

        # In QDQ format the op stays float in the graph but is fused into an
        # integer kernel at session init — and only when it is surrounded on
        # BOTH sides, DequantizeLinear -> op -> QuantizeLinear. Checking only the
        # input side counts an excluded node as quantized, because it still
        # consumes the DQ that closes the previous node's pair. That would
        # inflate the one number meant to reveal "smaller but not faster".
        fed_by_dequant = any(
            producers.get(tensor) is not None and producers[tensor].op_type == "DequantizeLinear"
            for tensor in node.input
        )
        feeds_quantize = any(
            consumer.op_type == "QuantizeLinear"
            for tensor in node.output
            for consumer in consumers.get(tensor, [])
        )
        if fed_by_dequant and feeds_quantize:
            quantized[base] = quantized.get(base, 0) + 1
        else:
            unquantized[base] = unquantized.get(base, 0) + 1

    return quantized, unquantized


def _activation_type() -> QuantType:
    """S8S8 on ARM, U8S8 on x86 — each is the combination its kernels prefer."""
    machine = host_mod.probe().machine.lower()
    return QuantType.QInt8 if machine.startswith(("arm", "aarch")) else QuantType.QUInt8


def to_fp16(src: Path, dst: Path) -> Path:
    """Halve the weights without touching the graph's numerics elsewhere.

    IO stays float32 so callers do not have to cast anything; only the internal
    weights and activations become float16.
    """
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(src))
    converted = convert_float_to_float16(model, keep_io_types=True, disable_shape_infer=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, str(dst))
    return dst


def to_int8(
    src: Path,
    dst: Path,
    calibration_loader,
    config: QuantConfig,
    input_name: str = "input",
) -> Path:
    """Static INT8 in QDQ format with per-channel weight scales.

    Per-channel matters: one scale for an entire conv weight tensor is set by its
    single largest-magnitude filter, so every other filter loses resolution. A
    scale per output channel costs nothing at inference and is usually the
    difference between "INT8 is fine" and "INT8 lost 5 points".
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    reader = LoaderCalibrationReader(calibration_loader, input_name, config.calibration_batches)
    host = host_mod.probe()
    reduce_range = host.reduce_range if config.reduce_range is None else config.reduce_range

    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        # Shape inference + graph cleanup first; the quantizer needs known shapes
        # to place QDQ pairs, and skipping this is the usual cause of large
        # unexplained accuracy drops.
        quant_pre_process(str(src), str(prepared), skip_symbolic_shape=False)

        quantize_static(
            model_input=str(prepared),
            model_output=str(dst),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            per_channel=config.per_channel,
            reduce_range=reduce_range,
            activation_type=_activation_type(),
            weight_type=QuantType.QInt8,
            calibrate_method=CALIBRATION_METHODS[config.calibrate],
            nodes_to_exclude=list(config.exclude_nodes) or None,
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    return dst


def quantize(
    src: str | Path,
    dst: str | Path,
    calibration_loader=None,
    config: QuantConfig | None = None,
    input_name: str = "input",
) -> QuantResult:
    """Quantize an ONNX artifact, choosing precision from the host when 'auto'."""
    config = config or QuantConfig()
    src, dst = Path(src), Path(dst)
    host = host_mod.probe()
    warnings: list[str] = []

    precision = config.precision
    if precision == "auto":
        precision = host.recommended_precision
        warnings.append(f"precision auto-selected: {precision} — {host.precision_rationale()}")
    elif precision == "int8" and host.int8_speed != "fast":
        message = f"INT8 requested but {host.precision_rationale()}"
        if not config.allow_slow_int8:
            raise RuntimeError(message + " Pass allow_slow_int8=True to proceed anyway.")
        warnings.append(message)

    if precision == "int8":
        if calibration_loader is None:
            raise ValueError("INT8 static quantization needs a calibration loader (use the train split)")
        to_int8(src, dst, calibration_loader, config, input_name=input_name)
    elif precision == "fp16":
        to_fp16(src, dst)
        warnings.append(
            "FP16 is a size win, not a speed win, on the ORT CPU provider — it casts back to "
            "float32 to compute. Expect roughly half the file and roughly unchanged latency."
        )
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)

    quantized_ops, unquantized_ops = _op_coverage(dst)
    if precision == "int8" and unquantized_ops:
        warnings.append(
            f"{sum(unquantized_ops.values())} compute op(s) stayed float32 "
            f"({', '.join(sorted(unquantized_ops))}) — the file shrank but those ops did not get faster."
        )

    return QuantResult(
        path=dst,
        precision=precision,
        size_mb=size_mb(dst),
        quantized_ops=quantized_ops,
        unquantized_ops=unquantized_ops,
        warnings=warnings,
    )
