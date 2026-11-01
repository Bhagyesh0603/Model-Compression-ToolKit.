"""Mixed-precision helpers shared by every GPU training loop.

fp16 autocast + a loss scaler roughly halves activation memory and uses the T4's
tensor cores. It is CUDA-only: on cpu/mps every helper here is a no-op, so
callers never branch on the device themselves.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn


def amp_enabled(device: str, requested: bool = True) -> bool:
    return bool(requested) and str(device).startswith("cuda")


def autocast(device: str, enabled: bool = True):
    """fp16 autocast context on CUDA, a null context everywhere else."""
    if amp_enabled(device, enabled):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


class Scaler:
    """GradScaler on CUDA with AMP, plain backward/step otherwise."""

    def __init__(self, device: str, enabled: bool = True):
        self.active = amp_enabled(device, enabled)
        self._scaler = torch.amp.GradScaler("cuda") if self.active else None

    def backward(self, loss: torch.Tensor) -> None:
        (self._scaler.scale(loss) if self.active else loss).backward()

    def unscale_(self, optimizer) -> None:
        if self.active:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer) -> None:
        if self.active:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()


def prepare_model(model: nn.Module, device: str) -> nn.Module:
    """Move to `device`; channels_last on CUDA, where it speeds up convolutions."""
    model.to(device)
    if str(device).startswith("cuda"):
        model.to(memory_format=torch.channels_last)
    return model


def prepare_batch(images: torch.Tensor, device: str) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    if str(device).startswith("cuda") and images.dim() == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    return images
