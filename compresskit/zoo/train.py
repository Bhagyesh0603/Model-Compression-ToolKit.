"""Baseline FP32 training, and checkpoint I/O for compressed models.

This produces the model everything else is measured against: the accuracy target
to preserve, and the teacher the distillation stage distills from.

Checkpoints pickle the whole module rather than a state_dict, because a pruned
model's layer widths no longer match any constructor signature — there is no
`resnet20()` that rebuilds a network whose third stage is 48 channels wide.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..core.amp import Scaler, autocast, prepare_batch, prepare_model
from ..core.benchmark import accuracy_torch
from ..core.host import training_device
from ..distillation import cosine_lr


@dataclass
class TrainConfig:
    epochs: int = 60
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 1
    label_smoothing: float = 0.0
    grad_clip: float = 0.0
    device: str | None = None
    amp: bool = True          # fp16 autocast on CUDA, no-op elsewhere
    accum_steps: int = 1      # micro-batches per optimizer step

    def __post_init__(self):
        if self.accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        if self.device is None:
            self.device = training_device()
        self.warmup_epochs = min(self.warmup_epochs, max(self.epochs - 1, 0))


def train(
    model: nn.Module,
    loader,
    config: TrainConfig | None = None,
    val_loader=None,
    on_epoch=None,
) -> nn.Module:
    """Train `model` from its current weights. Returns it on CPU in eval mode."""
    config = config or TrainConfig()
    device = config.device
    prepare_model(model, device).train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )

    scaler = Scaler(device, config.amp)
    accum = config.accum_steps
    batches = len(loader)
    for epoch in range(config.epochs):
        started = time.perf_counter()
        model.train()
        # On-device accumulators, read once per epoch (a per-batch .item() is a GPU sync).
        running_loss = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        seen = 0

        for index, (images, targets) in enumerate(loader):
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(
                    config.lr, epoch + index / max(batches, 1), config.epochs, config.warmup_epochs
                )

            images, targets = prepare_batch(images, device), targets.to(device, non_blocking=True)
            with autocast(device, config.amp):
                logits = model(images)
            logits = logits.float()
            loss = F.cross_entropy(logits, targets, label_smoothing=config.label_smoothing)

            if index % accum == 0:
                optimizer.zero_grad(set_to_none=True)
            scaler.backward(loss / accum)
            if (index + 1) % accum == 0 or index + 1 == batches:
                if config.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)

            running_loss += loss.detach() * targets.size(0)
            correct += (logits.argmax(1) == targets).sum()
            seen += targets.size(0)

        running_loss, correct = float(running_loss), float(correct)

        stats = {
            "epoch": epoch,
            "loss": running_loss / max(seen, 1),
            "train_acc": correct / max(seen, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
        }
        if val_loader is not None:
            stats["val_acc"] = accuracy_torch(model, val_loader, device=device)
            model.to(device)
        if on_epoch is not None:
            on_epoch(model, epoch, stats)

    return model.cpu().eval()


def save_model(model: nn.Module, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().eval(), "meta": meta or {}}, str(path))
    return path


def load_model(path: str | Path) -> tuple[nn.Module, dict]:
    """Load a checkpoint written by `save_model`.

    `weights_only=False` is required because the pickle holds the module object
    itself. Only load checkpoints you produced — this executes pickled code.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    return payload["model"].eval(), payload.get("meta", {})
