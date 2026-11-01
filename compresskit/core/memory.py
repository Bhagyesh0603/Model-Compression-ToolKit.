"""Find the largest training batch that fits the GPU, instead of guessing.

A batch that fits ResNet-20 OOMs ResNet-152 at 224x224 on a 16 GB T4, and nothing
used to catch it. `fit_batch_size` runs one real forward+backward (with the
teacher, when distilling) and halves on OOM. Callers pair the result with
gradient accumulation so the effective batch stays what was asked for.
"""

from __future__ import annotations

import gc

import torch
import torch.nn as nn

from .amp import Scaler, autocast, prepare_batch, prepare_model


def _is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _bn_buffers(*models: nn.Module | None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(buffer, clone) for every BatchNorm statistic in `models`, to put back after a probe."""
    saved = []
    for model in models:
        for module in (model.modules() if model is not None else ()):
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for buf in (module.running_mean, module.running_var, module.num_batches_tracked):
                    if buf is not None:
                        saved.append((buf, buf.detach().clone()))
    return saved


def fit_batch_size(
    model: nn.Module,
    input_shape: tuple[int, ...],
    start: int,
    device: str,
    teacher: nn.Module | None = None,
    amp: bool = True,
    minimum: int = 1,
) -> int:
    """Largest batch <= `start` whose fwd+bwd step fits on `device`.

    Returns `start` untouched off-CUDA: cpu/mps have no reliable OOM signal and
    the probe would only cost time. The model is left on `device` with its
    weights unchanged (the probe never steps an optimizer).
    """
    if not str(device).startswith("cuda"):
        return start

    was_training = model.training
    # Teacher first: callers may pass the model itself as its own teacher (it is both
    # during distillation), and the student pass must be the one left in train mode so
    # the probe sees train-mode memory (BN batch statistics, saved activations).
    if teacher is not None:
        prepare_model(teacher, device).eval()
    prepare_model(model, device).train()
    # The probe feeds random noise through train-mode BatchNorm, which would overwrite
    # the model's running statistics; snapshot them and put them back.
    saved = _bn_buffers(model, teacher)

    batch = max(start, minimum)
    try:
        while True:
            try:
                images = prepare_batch(torch.randn(batch, *input_shape[1:]), device)
                with autocast(device, amp):
                    if teacher is not None:
                        with torch.no_grad():
                            teacher(images)
                    logits = model(images)
                # Backward on a scalar of the right graph, scaled like real training.
                Scaler(device, amp).backward(logits.float().logsumexp(1).mean())
                break
            except BaseException as error:  # noqa: BLE001 - re-raised unless it is an OOM
                if not _is_oom(error) or batch <= minimum:
                    raise
                batch = max(batch // 2, minimum)
            finally:
                model.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        with torch.no_grad():
            for buf, original in saved:
                buf.copy_(original)
        model.train(was_training)
    return batch


def split_batch(requested: int, fitted: int) -> tuple[int, int]:
    """(micro_batch, accum_steps) that reproduce `requested` using batches of at most `fitted`."""
    micro = max(min(requested, fitted), 1)
    accum = max(-(-requested // micro), 1)
    return micro, accum
