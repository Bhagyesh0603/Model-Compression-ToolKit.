"""Torch -> ONNX, the single artifact format everything downstream measures.

One exported file serves both the latency benchmark (batch 1) and the accuracy
eval (large batch), so the two numbers always describe the same graph and the
same weights. Only the batch axis is symbolic; spatial dims stay static because
ORT's quantizer and its CPU kernels both do better with fixed shapes.
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

OPSET = 13


def to_onnx(
    model: torch.nn.Module,
    path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    opset: int = OPSET,
    dynamic_batch: bool = True,
) -> Path:
    """Export `model` to ONNX at `path` and return the path.

    The model is deep-copied to CPU/eval first so exporting can never mutate or
    leave device-resident state on the caller's model mid-pipeline.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    net = copy.deepcopy(model).cpu().eval()
    example = torch.randn(*input_shape)

    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

    with torch.no_grad():
        torch.onnx.export(
            net,
            example,
            str(path),
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            dynamo=False,
        )
    return path


def torch_name_from_onnx_node(node_name: str) -> str:
    """Recover the torch module path from an ONNX node's scope name.

    The exporter writes nested scopes like ``/layer2/layer2.0/downsample/
    downsample.0/Conv``, where each segment is named relative to its parent and
    consecutive segments overlap. Collapsing that overlap gives back
    ``layer2.0.downsample.0`` — which is what lets a sensitivity report name a
    layer the user can actually pass back in as an override.
    """
    # Only scoped names ("/a/b/OpType") carry a module path. A bare name like
    # "Conv_0" comes from a functional call with no owning module, so there is
    # nothing to recover — returning the op type as if it were a layer name
    # would invent a layer that the model does not have.
    if not node_name.startswith("/"):
        return ""

    segments = [s for s in node_name.split("/") if s][:-1]  # drop the trailing op type

    parts: list[str] = []
    for segment in segments:
        if parts and segment.startswith(parts[-1] + "."):
            parts[-1] = segment
        else:
            parts.append(segment)
    return ".".join(parts)


def layer_node_map(path: str | Path, op_types=("Conv", "Gemm", "MatMul", "ConvTranspose")) -> dict[str, list[str]]:
    """torch module path -> the ONNX node names it produced."""
    import onnx

    model = onnx.load(str(path))
    mapping: dict[str, list[str]] = {}
    for node in model.graph.node:
        if node.op_type not in op_types or not node.name:
            continue
        torch_name = torch_name_from_onnx_node(node.name)
        if torch_name:
            mapping.setdefault(torch_name, []).append(node.name)
    return mapping


def size_mb(path: str | Path) -> float:
    """Actual bytes on disk — never an estimate from parameter counts."""
    return Path(path).stat().st_size / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
