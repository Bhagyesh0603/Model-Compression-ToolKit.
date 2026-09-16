"""Quantization-aware training: recover accuracy plain PTQ alone loses.

Post-training quantization (`quantize.py`) rounds weights once, after training
already finished -- the model never gets a chance to compensate. QAT simulates
that same rounding *during* training, so gradients can push the full-precision
weights toward values that survive the rounding better.

Weight-only, matching the scope of `sensitivity.analyze`'s probe -- activation
quantization and AdaRound-style calibration are a separate, larger effort.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.host import training_device
from ..distillation.distill import cosine_lr


def fake_quantize_ste(weight: torch.Tensor, bits: int = 8, per_channel: bool = True) -> torch.Tensor:
    """Differentiable version of `sensitivity.analysis.fake_quantize`'s math.

    `.round()` has zero gradient everywhere, which is fine for `analyze()`'s
    `@torch.no_grad()` probe but would silently stop QAT's backward pass from
    ever reaching the weight it's supposed to be training. The straight-through
    estimator forwards the rounded value but backprops as if rounding were the
    identity function -- the standard trick this technique is named for.
    """
    qmax = 2 ** (bits - 1) - 1
    if per_channel and weight.dim() > 1:
        reduce_dims = tuple(range(1, weight.dim()))
        scale = weight.abs().amax(dim=reduce_dims, keepdim=True) / qmax
    else:
        scale = weight.abs().max() / qmax
    scale = scale.clamp(min=1e-12)
    quantized = (weight / scale).round().clamp(-qmax - 1, qmax) * scale
    return weight + (quantized - weight).detach()


def _quantized_conv_forward(bits: int, per_channel: bool):
    def forward(self, x):
        weight = fake_quantize_ste(self.weight, bits=bits, per_channel=per_channel)
        return self._conv_forward(x, weight, self.bias)

    return forward


def _quantized_linear_forward(bits: int, per_channel: bool):
    def forward(self, x):
        weight = fake_quantize_ste(self.weight, bits=bits, per_channel=per_channel)
        return F.linear(x, weight, self.bias)

    return forward


@contextmanager
def _fake_quantized(model: nn.Module, bits: int = 8, per_channel: bool = True):
    """Temporarily patch every Conv2d/Linear's forward to simulate quantization.

    Always restored on exit, exception or not -- a forward left patched would
    make a later `to_onnx` export silently trace the fake-quantized graph
    instead of the real one, the same restore-guarantee `distill()` applies to
    its borrowed teacher state.
    """
    originals = []
    try:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                originals.append((module, module.forward))
                module.forward = _quantized_conv_forward(bits, per_channel).__get__(module)
            elif isinstance(module, nn.Linear):
                originals.append((module, module.forward))
                module.forward = _quantized_linear_forward(bits, per_channel).__get__(module)
        yield model
    finally:
        for module, original in originals:
            module.forward = original


@dataclass
class QATConfig:
    epochs: int = 3
    lr: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 1
    grad_clip: float = 0.0
    device: str | None = None
    bits: int = 8
    per_channel: bool = True

    def __post_init__(self):
        if self.device is None:
            self.device = training_device()
        # A short run must not be entirely warmup, or the LR ramps up and the
        # schedule ends before it ever reaches the rate you asked for.
        self.warmup_epochs = min(self.warmup_epochs, max(self.epochs - 1, 0))


def qat(
    model: nn.Module,
    loader,
    config: QATConfig | None = None,
    on_epoch=None,
) -> nn.Module:
    """Fine-tune `model` with simulated weight quantization noise in the forward pass.

    Trains in place and returns the model. `on_epoch(model, epoch, stats)` runs
    after each epoch -- use it to evaluate or checkpoint. No teacher: this
    trains against ground-truth labels only, unlike `distill()`.
    """
    config = config or QATConfig()
    device = config.device

    if len(loader) == 0:
        raise ValueError(
            "QAT needs at least one training batch -- got an empty loader; a silently "
            "completed 0-epoch run would report success while never having updated a "
            "single weight"
        )

    model.to(device).train()

    with _fake_quantized(model, bits=config.bits, per_channel=config.per_channel):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=True,
        )

        batches = len(loader)
        for epoch in range(config.epochs):
            totals = {"loss": 0.0, "correct": 0, "seen": 0}

            for index, (images, targets) in enumerate(loader):
                for group in optimizer.param_groups:
                    group["lr"] = cosine_lr(
                        config.lr, epoch + index / max(batches, 1), config.epochs, config.warmup_epochs
                    )

                images, targets = images.to(device), targets.to(device)
                logits = model(images)
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if config.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

                count = targets.size(0)
                totals["loss"] += loss.item() * count
                totals["correct"] += int((logits.argmax(1) == targets).sum())
                totals["seen"] += count

            seen = max(totals["seen"], 1)
            stats = {
                "epoch": epoch,
                "loss": totals["loss"] / seen,
                "train_acc": totals["correct"] / seen,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if on_epoch is not None:
                on_epoch(model, epoch, stats)

    return model.cpu().eval()
