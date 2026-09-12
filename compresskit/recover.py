"""BatchNorm re-estimation — the free accuracy recovery step after pruning.

Removing channels changes the distribution of every downstream activation, but
BN's `running_mean`/`running_var` still describe the *old* network. At eval time
those stale statistics are applied as a fixed affine shift, so the model can lose
several points of accuracy for a reason that has nothing to do with lost
capacity.

Fixing it costs no backprop and no labels: reset the buffers and recompute them
from a few hundred forward passes over calibration data. Always run this before
judging how much accuracy pruning actually cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _bn_layers(model: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    return [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    loader,
    batches: int | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Recompute BN running statistics in place from `loader`. Returns the model.

    `loader` must be calibration data drawn from the train split. Using test data
    here would leak it into the model's fitted parameters and make the reported
    accuracy meaningless.
    """
    layers = _bn_layers(model)
    if not layers:
        return model

    was_training = model.training
    model.to(device)

    saved_momentum = []
    for bn in layers:
        saved_momentum.append(bn.momentum)
        bn.reset_running_stats()
        # momentum=None switches BN to a cumulative average, so the result is the
        # exact mean/var over everything we feed it rather than an EMA weighted
        # towards whichever batch happened to come last.
        bn.momentum = None

    model.train()
    for index, batch in enumerate(loader):
        if batches is not None and index >= batches:
            break
        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        model(images.to(device))

    for bn, momentum in zip(layers, saved_momentum):
        bn.momentum = momentum

    model.train(was_training)
    return model
