"""Per-layer sensitivity: which layer is actually costing you accuracy.

Quantize one layer at a time, leave the rest in float, and measure how far the
model's output logits move. A layer whose 8-bit version barely perturbs the
output is safe to compress hard; a layer that wrecks the output is the one to
protect. That ranking is what turns "accuracy dropped 6 points" into "accuracy
dropped 6 points *because of these two layers*", and it drives both the
quantization exclusion list and the per-layer pruning ratios.

Cost is O(layers) forward passes over a few calibration batches — no backprop,
no labels, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .export import layer_node_map

QUANTIZABLE = (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)


def fake_quantize(weight: torch.Tensor, bits: int = 8, per_channel: bool = True) -> torch.Tensor:
    """Symmetric quantize-dequantize, mirroring what the ORT export will do.

    Per-output-channel by default: a single scale for the whole tensor is set by
    its largest filter, so every smaller filter is left using a fraction of the
    integer range.
    """
    qmax = 2 ** (bits - 1) - 1
    if per_channel and weight.dim() > 1:
        reduce_dims = tuple(range(1, weight.dim()))
        scale = weight.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    else:
        scale = weight.abs().max() / qmax
    scale = scale.clamp(min=1e-12)
    return (weight / scale).round().clamp(-qmax - 1, qmax) * scale


@dataclass
class LayerSensitivity:
    name: str
    kind: str
    parameters: int
    mse: float
    sqnr_db: float

    def summary(self) -> str:
        return f"{self.name:<34} {self.kind:<7} {self.parameters:>9,}  SQNR {self.sqnr_db:7.2f} dB  MSE {self.mse:.3e}"


@dataclass
class SensitivityReport:
    """Layers ranked most-damaging first."""

    layers: list[LayerSensitivity]
    bits: int
    batches: int

    def ranked(self) -> list[LayerSensitivity]:
        return sorted(self.layers, key=lambda layer: layer.sqnr_db)

    def most_sensitive(self, k: int = 2) -> list[str]:
        return [layer.name for layer in self.ranked()[:k]]

    def exclude_nodes(self, onnx_path, k: int = 2) -> list[str]:
        """ONNX node names for the k most sensitive layers.

        Feed straight into ``QuantConfig.exclude_nodes`` to keep those layers in
        float while everything else goes to INT8.
        """
        mapping = layer_node_map(onnx_path)
        nodes: list[str] = []
        for name in self.most_sensitive(k):
            found = mapping.get(name)
            if not found:
                # Returning [] here would hand the quantizer an empty exclusion
                # list, which it treats as "exclude nothing" — the protection
                # would silently not happen and the accuracy loss would look
                # inexplicable.
                raise KeyError(
                    f"layer {name!r} has no matching node in {onnx_path}; "
                    f"known layers: {sorted(mapping)[:8]}"
                )
            nodes.extend(found)
        return nodes

    def prune_ratios(self, base_ratio: float, protect: int = 2, protect_scale: float = 0.5) -> dict[str, float]:
        """Per-layer pruning ratios that prune sensitive layers less.

        Uniform ratios treat a fragile early layer the same as a redundant wide
        one; this scales the ratio down for whichever layers the ranking says are
        carrying the most signal. The classifier is skipped — it is never pruned,
        so naming it here would just be an override that goes nowhere.
        """
        classifier = next(
            (layer.name for layer in reversed(self.layers) if layer.kind == "Linear"),
            None,
        )
        protected = [name for name in self.most_sensitive(protect + 1) if name != classifier][:protect]
        return {name: base_ratio * protect_scale for name in protected}

    def summary(self, top: int = 10) -> str:
        ranked = self.ranked()
        lines = [f"per-layer {self.bits}-bit sensitivity (lower SQNR = more damage, worst first):"]
        lines += [f"  {layer.summary()}" for layer in ranked[:top]]
        if len(ranked) > top:
            lines.append(f"  ... {len(ranked) - top} more")
        return "\n".join(lines)


def _sqnr_db(reference: torch.Tensor, perturbed: torch.Tensor) -> float:
    noise = (reference - perturbed).pow(2).sum().item()
    signal = reference.pow(2).sum().item()
    if noise <= 0:
        return float("inf")
    if signal <= 0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(signal / noise)).item()


@torch.no_grad()
def analyze(
    model: nn.Module,
    loader,
    bits: int = 8,
    batches: int = 4,
    device: str = "cpu",
    per_channel: bool = True,
) -> SensitivityReport:
    """Rank every quantizable layer by the output error its quantization causes."""
    was_training = model.training
    model.eval().to(device)

    inputs = []
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        inputs.append(images.to(device))
    if not inputs:
        raise ValueError("sensitivity analysis needs at least one calibration batch")

    reference = torch.cat([model(x) for x in inputs])

    results = []
    for name, module in model.named_modules():
        if not isinstance(module, QUANTIZABLE):
            continue

        original = module.weight.data
        module.weight.data = fake_quantize(original, bits=bits, per_channel=per_channel)
        try:
            perturbed = torch.cat([model(x) for x in inputs])
        finally:
            module.weight.data = original

        results.append(
            LayerSensitivity(
                name=name,
                kind=type(module).__name__.replace("2d", ""),
                parameters=sum(p.numel() for p in module.parameters()),
                mse=(reference - perturbed).pow(2).mean().item(),
                sqnr_db=_sqnr_db(reference, perturbed),
            )
        )

    model.train(was_training)
    return SensitivityReport(layers=results, bits=bits, batches=len(inputs))
