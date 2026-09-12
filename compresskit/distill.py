"""Compression-aware knowledge distillation.

The teacher is the *original uncompressed model* — not a separately trained
larger network. It already exists, it is already the best version of this exact
function, and its soft logits carry the inter-class structure the pruned student
just lost. That removes the usual cost of KD (training and storing a teacher)
and makes distillation a natural recovery step rather than a separate project.

Two losses:

* ``kd`` — the classic blend, ``(1-a)*CE + a*T^2*KL``. Safe default.
* ``dkd`` — Decoupled KD. Splits the KL into "is it the target class" (TCKD) and
  "how are the wrong classes ranked" (NCKD), then weights them separately.
  Classic KD implicitly couples the two and suppresses NCKD exactly when the
  teacher is confident, which is when that ranking is most informative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.host import training_device


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """T^2-scaled KL(student || teacher) at temperature T.

    The T^2 factor keeps gradient magnitude comparable across temperatures —
    softening the distributions shrinks the gradients by roughly 1/T^2.
    """
    student_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temperature**2)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Decoupled KD: alpha * TCKD + beta * NCKD."""
    target_mask = F.one_hot(targets, student_logits.size(1)).bool()

    student_prob = F.softmax(student_logits / temperature, dim=1)
    teacher_prob = F.softmax(teacher_logits / temperature, dim=1)

    # TCKD — a two-way distribution: P(target) vs P(everything else).
    # Clamped because a confident student drives the target probability to
    # exactly 1.0 in float32, making the non-target term 0 and log(0) = -inf.
    # That poisons the loss to NaN, the weights follow, and the pipeline then
    # reports chance accuracy as a measured result.
    def binary(prob):
        target_p = prob[target_mask].unsqueeze(1).clamp(min=1e-7, max=1.0 - 1e-7)
        return torch.cat([target_p, 1.0 - target_p], dim=1)

    tckd = F.kl_div(binary(student_prob).log(), binary(teacher_prob), reduction="batchmean") * (temperature**2)

    # NCKD — the ranking among non-target classes, with the target masked out
    # *before* the softmax so the remaining classes renormalise to 1.
    masked_student = student_logits / temperature - 1000.0 * target_mask
    masked_teacher = teacher_logits / temperature - 1000.0 * target_mask
    nckd = F.kl_div(
        F.log_softmax(masked_student, dim=1),
        F.softmax(masked_teacher, dim=1),
        reduction="batchmean",
    ) * (temperature**2)

    return alpha * tckd + beta * nckd


@dataclass
class DistillConfig:
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    loss: str = "kd"
    temperature: float = 4.0
    # kd: blend weight on the KL term. dkd: weight on TCKD. Defaulted per loss.
    alpha: float | None = None
    beta: float = 8.0
    ce_weight: float = 1.0
    warmup_epochs: int = 1
    grad_clip: float = 0.0
    device: str | None = None

    def __post_init__(self):
        if self.loss not in ("kd", "dkd"):
            raise ValueError(f"unknown loss {self.loss!r}; choose 'kd' or 'dkd'")
        if self.alpha is None:
            self.alpha = 0.3 if self.loss == "kd" else 1.0
        if self.device is None:
            self.device = training_device()
        # A short run must not be entirely warmup, or the LR ramps up and the
        # schedule ends before it ever reaches the rate you asked for.
        self.warmup_epochs = min(self.warmup_epochs, max(self.epochs - 1, 0))


def cosine_lr(base_lr: float, step: float, total_epochs: int, warmup_epochs: int = 0) -> float:
    """Linear warmup then cosine decay. `step` is a fractional epoch index.

    Warmup matters more than usual here: a freshly pruned model starts from a
    damaged optimum, and a full-size first step can push it somewhere it never
    recovers from.
    """
    if warmup_epochs and step < warmup_epochs:
        return base_lr * (step + 1e-8) / warmup_epochs
    span = max(total_epochs - warmup_epochs, 1e-8)
    decayed = (step - warmup_epochs) / span
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(decayed, 1.0)))


def distill(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    config: DistillConfig | None = None,
    on_epoch=None,
) -> nn.Module:
    """Fine-tune `student` against `teacher` logits + ground-truth labels.

    Trains in place and returns the student. `on_epoch(student, epoch, stats)`
    runs after each epoch — use it to evaluate or checkpoint.
    """
    config = config or DistillConfig()
    device = config.device

    student.to(device).train()
    # The teacher is only ever read from, but it belongs to the caller — freeze
    # it for the duration and hand it back exactly as it arrived, or a direct
    # `distill(...)` call leaves the caller's model permanently untrainable.
    teacher_was_training = teacher.training
    teacher_grads = [p.requires_grad for p in teacher.parameters()]
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=config.lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        nesterov=True,
    )

    batches = len(loader)
    for epoch in range(config.epochs):
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "correct": 0, "seen": 0}

        for index, (images, targets) in enumerate(loader):
            for group in optimizer.param_groups:
                group["lr"] = cosine_lr(
                    config.lr, epoch + index / max(batches, 1), config.epochs, config.warmup_epochs
                )

            images, targets = images.to(device), targets.to(device)

            with torch.no_grad():
                teacher_logits = teacher(images)
            student_logits = student(images)

            ce = F.cross_entropy(student_logits, targets)
            if config.loss == "kd":
                soft = kd_loss(student_logits, teacher_logits, config.temperature)
                loss = (1.0 - config.alpha) * ce + config.alpha * soft
            else:
                soft = dkd_loss(
                    student_logits, teacher_logits, targets, config.temperature, config.alpha, config.beta
                )
                loss = config.ce_weight * ce + soft

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            optimizer.step()

            count = targets.size(0)
            totals["loss"] += loss.item() * count
            totals["ce"] += ce.item() * count
            totals["kd"] += soft.item() * count
            totals["correct"] += int((student_logits.argmax(1) == targets).sum())
            totals["seen"] += count

        seen = max(totals["seen"], 1)
        stats = {
            "epoch": epoch,
            "loss": totals["loss"] / seen,
            "ce": totals["ce"] / seen,
            "kd": totals["kd"] / seen,
            "train_acc": totals["correct"] / seen,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if on_epoch is not None:
            on_epoch(student, epoch, stats)

    for param, required in zip(teacher.parameters(), teacher_grads):
        param.requires_grad_(required)
    teacher.train(teacher_was_training)

    return student.cpu().eval()
