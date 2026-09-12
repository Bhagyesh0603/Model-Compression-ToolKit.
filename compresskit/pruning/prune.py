"""Structured channel pruning via Torch-Pruning's dependency graph.

Structured, not unstructured: zeroing individual weights leaves the tensor the
same size, and no maintained CPU runtime turns that sparsity into speed. Removing
whole channels shrinks the actual GEMM, which is where CPU latency comes from.

Three things this wrapper adds over calling Torch-Pruning directly:

* **SIMD alignment computed per layer, not as one global knob.** Torch-Pruning's
  own `round_to` is a single value for the whole model, and it only floors the
  kept-channel count down to the nearest multiple — never rounds to the
  nearest one. On a layer whose width equals one alignment block (e.g. 16
  channels at align=16), that floor lands on 0, Torch-Pruning rejects the
  group as invalid, and the whole layer is *silently* left at 100% of its
  channels. The usual "fix" is relaxing `round_to` for the entire model, but
  that throws away the alignment (and the SIMD win it exists for) on every
  *other* layer too, just to accommodate the one narrow one. Instead, this
  wrapper computes, per layer, the nearest achievable multiple of the
  requested alignment — only shrinking the alignment used for a specific
  layer when that layer's own width can't fit two aligned values at its
  ratio, and leaving every other layer at full alignment. Any layer that had
  to shrink, or that turned out truly unprunable at any alignment, is named in
  `PruneStats.warnings`.
* **Ratio overrides resolved across shared dependency groups.** A residual
  connection ties every layer feeding the same addition into one Torch-Pruning
  group that must share a single ratio. If two of those layers get different
  `ratio_overrides` (which sensitivity-driven protection can produce), Torch-
  Pruning silently keeps whichever one it iterates to first. This wrapper
  detects that and resolves it to the gentlest (most protective) ratio in the
  group instead, and records the resolution in `PruneStats.warnings`.
* **The classifier is never pruned.** Its output width is the class count, so
  pruning it changes what the model predicts rather than how fast it predicts.

Untested territory: grouped/depthwise convolutions (MobileNet-style
`groups>1`). None of `compresskit.zoo.models`'s architectures use them, so the
grouping and alignment logic above has never been exercised against a layer
whose channels aren't independently prunable one at a time.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_pruning as tp

CRITERIA = {
    "magnitude": lambda: tp.importance.GroupMagnitudeImportance(p=1),
    "magnitude_l2": lambda: tp.importance.GroupMagnitudeImportance(p=2),
    "taylor": lambda: tp.importance.GroupTaylorImportance(),
    "random": lambda: tp.importance.RandomImportance(),
}


@dataclass
class PruneConfig:
    """`global_pruning=True` is not protected by the per-layer SIMD-alignment
    logic below — it distributes pruning unevenly across layers by design, so
    a fixed per-layer ratio can't be computed ahead of time the way the
    default (local) mode allows. A layer can still end up silently untouched
    there; `StructuredPruner.stats()` only warns about it after the fact
    rather than preventing it.
    """

    ratio: float = 0.5
    align: int = 16
    steps: int = 1
    criterion: str = "magnitude"
    global_pruning: bool = False
    max_ratio: float = 0.9
    ratio_overrides: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.criterion not in CRITERIA:
            raise ValueError(f"unknown criterion {self.criterion!r}; choose from {sorted(CRITERIA)}")
        if not 0.0 <= self.ratio < 1.0:
            raise ValueError(f"ratio must be in [0, 1), got {self.ratio}")
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")


@dataclass
class PruneStats:
    params_before: int
    params_after: int
    macs_before: int
    macs_after: int
    ratio_requested: float
    align: int
    align_effective: int
    steps: int
    channels: dict[str, tuple[int, int]]
    warnings: list[str] = field(default_factory=list)

    @property
    def param_drop(self) -> float:
        return 1.0 - self.params_after / self.params_before if self.params_before else 0.0

    @property
    def mac_drop(self) -> float:
        return 1.0 - self.macs_after / self.macs_before if self.macs_before else 0.0

    def summary(self) -> str:
        lines = [
            f"params {self.params_before:,} -> {self.params_after:,} ({self.param_drop * 100:.1f}% removed)  "
            f"MACs {self.macs_before / 1e6:.1f}M -> {self.macs_after / 1e6:.1f}M ({self.mac_drop * 100:.1f}% removed)"
        ]
        for warning in self.warnings:
            lines.append(f"! {warning}")
        return "\n".join(lines)


def classifier_layers(model: nn.Module) -> list[nn.Module]:
    """The output layer whose width is the class count — never prunable.

    Falls back to the last conv for fully-convolutional heads. Returning nothing
    would leave the output layer unprotected, and pruning it changes how many
    classes the model can predict.
    """
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if linears:
        return linears[-1:]
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    return convs[-1:] if convs else []


def _aligned_ratio(width: int, ratio: float, align: int) -> tuple[float, int]:
    """The per-layer ratio that lands the kept-channel count on the nearest
    achievable multiple of `align` for this layer's own width, plus the
    alignment actually used to get there.

    Torch-Pruning's `round_to` floors the kept count to the nearest multiple
    of a single global alignment; if a layer's width only has room for one
    aligned value (width // align < 2), that floor lands on 0, the group gets
    rejected as invalid, and the layer silently keeps every channel. Rounding
    to *nearest* instead of flooring, and shrinking the alignment only for the
    specific layer that needs it, fixes both problems: the result is closer to
    the requested ratio on average, and a narrow layer no longer forces a
    weaker alignment onto layers that didn't need any relaxation at all.
    """
    if width <= 0 or ratio <= 0:
        return 0.0, align

    layer_align = align if align and align > 1 else 1
    while layer_align > 1 and width // layer_align < 2:
        layer_align //= 2

    if layer_align <= 1:
        # No alignment is achievable while keeping at least one channel — the
        # ratio applies directly, clamped so the layer is never fully removed.
        kept = max(1, min(width, round(width * (1 - ratio))))
        return 1 - kept / width, layer_align

    desired_kept = width * (1 - ratio)
    max_blocks = width // layer_align
    n_blocks = min(max_blocks, max(1, round(desired_kept / layer_align)))
    kept = n_blocks * layer_align
    return 1 - kept / width, layer_align


def _dependency_groups(
    model: nn.Module, example_inputs: torch.Tensor, ignored: list[nn.Module]
) -> list[list[str]]:
    """Every set of layer names Torch-Pruning forces to share one channel
    dimension — e.g. a residual block's main path and its downsample
    shortcut, tied together across a whole stage — plus every layer with no
    ties at all, as a group of one.

    Both ratio-conflict resolution and per-layer alignment rounding need this
    same grouping: whatever ratio or alignment is decided has to be applied
    identically to every member of a group, or two tied layers can come out
    with different values and Torch-Pruning silently keeps only one of them.
    """
    dg = tp.dependency.DependencyGraph().build_dependency(model, example_inputs=example_inputs)
    name_of = {module: name for name, module in model.named_modules()}

    groups = []
    for group in dg.get_all_groups(ignored_layers=ignored, root_module_types=[nn.Conv2d, nn.Linear]):
        members = sorted(
            {
                name_of[dep.target.module]
                for dep, _ in group
                if dep.target.module in name_of and dg.is_out_channel_pruning_fn(dep.handler)
            }
        )
        if members:
            groups.append(members)
    return groups


def _resolve_group_overrides(
    groups: list[list[str]], ratio_overrides: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Expand ratio overrides to cover every layer a residual connection ties them to.

    Any member of a tied group left without an explicit override would
    otherwise fall back to the plain global ratio — but the group can only
    ever move as one, so that "no override" default silently competes with
    the override once anything downstream (like alignment, below) needs one
    definite ratio per layer. Every group with at least one override is
    therefore unified onto the gentlest (most protective) override present,
    covering every member whether or not it was named explicitly. Only when
    two *different* overrides land in the same group — a real conflict,
    since Torch-Pruning would otherwise apply whichever one it reaches first
    while iterating internal state and silently drop the other — does this
    also record a warning.
    """
    resolved = dict(ratio_overrides)
    notes: list[str] = []
    for members in groups:
        overridden = {name: ratio_overrides[name] for name in members if name in ratio_overrides}
        if not overridden:
            continue  # nothing in this group was overridden; the global ratio applies untouched

        gentlest = min(overridden.values())
        if len(set(overridden.values())) > 1:
            outvoted = sorted(name for name, r in overridden.items() if r != gentlest)
            notes.append(
                f"ratio_overrides conflict: {outvoted} share a residual/shortcut connection with "
                f"a layer overridden to {gentlest}, and Torch-Pruning can only give the whole tied "
                f"group {members} one ratio — using the gentlest requested ({gentlest}) instead of "
                f"silently picking one member's value"
            )
        for name in members:
            resolved[name] = gentlest
    return resolved, notes


class StructuredPruner:
    """Iterative structured pruning. Call `step()`, fine-tune, repeat.

    Gradual pruning with recovery between steps beats one-shot removal at the
    same final ratio: each small step perturbs the network little enough that
    fine-tuning can still find its way back.
    """

    def __init__(
        self,
        model: nn.Module,
        example_inputs: torch.Tensor,
        config: PruneConfig | None = None,
        ignored_layers: list[nn.Module] | None = None,
        loader=None,
        batches: int = 8,
    ):
        self.config = config or PruneConfig()
        if self.config.criterion == "taylor" and loader is None:
            raise ValueError(
                "criterion='taylor' ranks channels by how much removing them would change the "
                "loss, which needs a real forward+backward pass over labeled data — pass "
                "loader=<a DataLoader yielding (images, labels) batches> (e.g. the same "
                "calibration_loader used for BN recalibration and sensitivity analysis)"
            )
        self.loader = loader
        self.batches = batches

        # Torch-Pruning rewrites modules in place, and the unpruned model is
        # still needed as the distillation teacher, so we work on a copy.
        self.model = copy.deepcopy(model).cpu().eval()
        self.example_inputs = example_inputs.cpu()

        ignored = list(ignored_layers) if ignored_layers is not None else classifier_layers(self.model)

        by_name = dict(self.model.named_modules())
        for name in self.config.ratio_overrides:
            if by_name.get(name) is None:
                raise KeyError(f"ratio override names unknown layer {name!r}")

        groups = _dependency_groups(self.model, self.example_inputs, ignored)
        resolved_overrides, group_notes = _resolve_group_overrides(groups, self.config.ratio_overrides)
        self.warnings: list[str] = list(group_notes)

        ratio_dict = {by_name[name]: ratio for name, ratio in resolved_overrides.items()}

        self.macs_before, self.params_before = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        self._widths_before = self._channel_widths()

        def ratio_of(name: str) -> float:
            return resolved_overrides.get(name, self.config.ratio)

        self.ratio_of = ratio_of
        self._candidate_names = {name for name in self._widths_before if by_name[name] not in ignored}

        requested_align = self.config.align
        round_to_for_pruner = requested_align or None
        self.effective_align = requested_align
        self._flagged_unprunable: set[str] = set()
        # Per-layer alignment only makes sense against a known, fixed
        # per-layer ratio; global pruning distributes pruning unevenly across
        # layers by design, so it relies solely on the post-hoc check in
        # `stats()` instead, same as before.
        if requested_align and not self.config.global_pruning:
            # Torch-Pruning clamps whatever ratio we hand it to max_ratio
            # internally (`min(requested, max_pruning_ratio)`), *after* our
            # alignment math already ran — so aligning against the raw
            # requested ratio and letting the clamp happen afterwards would
            # silently reintroduce misalignment on any layer whose ratio
            # exceeds max_ratio. Capping here first means we align against
            # what will actually be applied. The result can still land up to
            # half an alignment block away from max_ratio itself — the same
            # tolerance already accepted for the primary requested ratio.
            def capped_ratio_of(name: str) -> float:
                return min(ratio_of(name), self.config.max_ratio)

            relaxed: dict[str, int] = {}
            unprunable: list[str] = []
            per_layer_ratio: dict[str, float] = {}
            for members in groups:
                prunable = [n for n in members if n in self._candidate_names]
                if not prunable:
                    continue
                # Tied members share one channel dimension by construction, so
                # they must all see the same width and the same resolved
                # ratio here — computing `_aligned_ratio` once for the whole
                # group (instead of once per name) is what keeps every member
                # landing on the *same* rounded value. Rounding independently
                # per name reintroduces the exact silent-desync bug the
                # group-conflict resolution above exists to prevent, just
                # self-inflicted by alignment instead of by the caller.
                widths = {self._widths_before[n] for n in prunable}
                ratios = {capped_ratio_of(n) for n in prunable}
                if len(widths) != 1 or len(ratios) != 1:
                    # Should not happen — resolved_overrides already unified
                    # the ratio, and tied modules share a channel dimension —
                    # but if it ever does, fall back to per-name rounding
                    # rather than guessing which value should win the group.
                    for n in prunable:
                        ratio = capped_ratio_of(n)
                        if ratio <= 0:
                            continue
                        aligned_ratio, layer_align = _aligned_ratio(self._widths_before[n], ratio, requested_align)
                        per_layer_ratio[n] = aligned_ratio
                        if layer_align != requested_align:
                            relaxed[n] = layer_align
                        if aligned_ratio <= 0:
                            unprunable.append(n)
                    continue

                ratio = ratios.pop()
                if ratio <= 0:
                    continue
                aligned_ratio, layer_align = _aligned_ratio(widths.pop(), ratio, requested_align)
                for n in prunable:
                    per_layer_ratio[n] = aligned_ratio
                if layer_align != requested_align:
                    for n in prunable:
                        relaxed[n] = layer_align
                if aligned_ratio <= 0:
                    unprunable.extend(prunable)

            if per_layer_ratio:
                if relaxed:
                    self.warnings.append(
                        f"SIMD alignment tightened per layer instead of relaxed for everyone: "
                        f"{dict(sorted(relaxed.items()))} use a smaller alignment than the "
                        f"requested {requested_align} because their own width can't fit two "
                        f"aligned blocks at that ratio — every other layer keeps the full "
                        f"{requested_align}-alignment"
                    )
                if unprunable:
                    self._flagged_unprunable = set(unprunable)
                    self.warnings.append(
                        f"{sorted(unprunable)} cannot be pruned at all while honoring their "
                        f"requested ratio at any alignment (width is too narrow to remove even "
                        f"one channel) — lower the ratio for these layers or add them to "
                        f"ignored_layers"
                    )
                for name, ratio in per_layer_ratio.items():
                    ratio_dict[by_name[name]] = ratio
                round_to_for_pruner = None  # alignment is now baked into each layer's own ratio
                self.effective_align = min(relaxed.values(), default=requested_align)

        self.pruner = tp.pruner.MetaPruner(
            self.model,
            self.example_inputs,
            importance=CRITERIA[self.config.criterion](),
            pruning_ratio=self.config.ratio,
            pruning_ratio_dict=ratio_dict or None,
            max_pruning_ratio=self.config.max_ratio,
            iterative_steps=self.config.steps,
            global_pruning=self.config.global_pruning,
            ignored_layers=ignored,
            round_to=round_to_for_pruner,
        )
        self.steps_taken = 0

    def _channel_widths(self) -> dict[str, int]:
        widths = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                widths[name] = module.out_channels
            elif isinstance(module, nn.Linear):
                widths[name] = module.out_features
        return widths

    @property
    def steps_remaining(self) -> int:
        return self.config.steps - self.steps_taken

    def _accumulate_gradients(self) -> None:
        """Populate `.grad` on every parameter so Taylor importance has
        something to read — it ranks channels by how much removing them
        would move a real loss, which needs a real backward pass, not just
        the weight magnitudes the other criteria use.

        Re-run before every `step()`, not just the first: the weights (and
        therefore the gradients) change after each pruning step, so a Taylor
        ranking computed against stale, pre-pruning weights would be
        importance-scoring a network that no longer exists.
        """
        self.model.zero_grad(set_to_none=True)
        for index, batch in enumerate(self.loader):
            if index >= self.batches:
                break
            if not (isinstance(batch, (list, tuple)) and len(batch) >= 2):
                raise ValueError(
                    f"criterion='taylor' needs (images, labels) batches for a real "
                    f"cross-entropy loss; got {type(batch).__name__} instead — a loader "
                    f"yielding bare images (as sensitivity.analyze tolerates) isn't enough"
                )
            images, labels = batch[0], batch[1]
            loss = F.cross_entropy(self.model(images), labels)
            loss.backward()

    def step(self) -> nn.Module:
        """Remove one increment of channels. Fine-tune after this, then repeat."""
        if self.steps_remaining <= 0:
            raise RuntimeError("all pruning steps already taken")
        if self.config.criterion == "taylor":
            self._accumulate_gradients()
        self.pruner.step()
        if self.config.criterion == "taylor":
            # Nothing downstream (a between_steps distillation fine-tune, in
            # particular) should see these as a head start on its own loss.
            self.model.zero_grad(set_to_none=True)
        self.steps_taken += 1
        return self.model

    def run(self, between_steps=None) -> nn.Module:
        """Take every step, calling `between_steps(model, step_index)` in between.

        `between_steps` is where recovery goes — BN re-estimation, or a short
        distillation fine-tune. Without it this degrades to one-shot pruning.
        """
        while self.steps_remaining > 0:
            self.step()
            if between_steps is not None and self.steps_remaining > 0:
                between_steps(self.model, self.steps_taken)
        return self.model

    def stats(self) -> PruneStats:
        macs_after, params_after = tp.utils.count_ops_and_params(self.model, self.example_inputs)
        widths_after = self._channel_widths()

        warnings = list(self.warnings)
        candidates = sorted(self._candidate_names - self._flagged_unprunable)
        if not self.config.global_pruning:
            for name in candidates:
                before = self._widths_before[name]
                after = widths_after.get(name, before)
                requested = self.ratio_of(name)
                if requested > 0 and after == before:
                    warnings.append(
                        f"{name}: requested ratio {requested:.3f} but 0 of {before} channels "
                        f"were actually removed — likely blocked by a dependency constraint "
                        f"(e.g. a residual connection) or a max_ratio cap"
                    )
        elif self.config.align and candidates:
            # global_pruning distributes pruning unevenly across layers by
            # design, so a per-layer "0% pruned" flag would mostly be noise
            # here — some layers are *supposed* to come out untouched. But
            # global mode also isn't protected by the per-layer alignment
            # logic above at all (see PruneConfig's docstring), so a
            # suspiciously large fraction of untouched layers is the
            # signature of SIMD alignment silently rejecting groups rather
            # than global importance legitimately sparing them.
            untouched = [n for n in candidates if widths_after.get(n, self._widths_before[n]) == self._widths_before[n]]
            if len(untouched) > len(candidates) / 2:
                warnings.append(
                    f"{len(untouched)} of {len(candidates)} candidate layers were left completely "
                    f"unpruned ({sorted(untouched)[:8]}{', ...' if len(untouched) > 8 else ''}) — "
                    f"global_pruning=True is not protected by this module's per-layer SIMD-alignment "
                    f"logic, so this may be align={self.config.align} silently rejecting groups rather "
                    f"than global importance legitimately sparing these layers; try align=1 to check"
                )

        return PruneStats(
            params_before=self.params_before,
            params_after=params_after,
            macs_before=self.macs_before,
            macs_after=macs_after,
            ratio_requested=self.config.ratio,
            align=self.config.align,
            align_effective=self.effective_align,
            steps=self.config.steps,
            channels={k: (self._widths_before[k], widths_after[k]) for k in widths_after},
            warnings=warnings,
        )


def prune(
    model: nn.Module,
    example_inputs: torch.Tensor,
    ratio: float = 0.5,
    align: int = 16,
    steps: int = 1,
    criterion: str = "magnitude",
    between_steps=None,
    loader=None,
    batches: int = 8,
    **kwargs,
) -> tuple[nn.Module, PruneStats]:
    """One-call structured pruning. Returns (pruned copy, stats).

    `loader` (yielding (images, labels) batches) is only required when
    `criterion="taylor"` — every other criterion ranks channels from weights
    alone and ignores it.
    """
    pruner = StructuredPruner(
        model,
        example_inputs,
        PruneConfig(ratio=ratio, align=align, steps=steps, criterion=criterion, **kwargs),
        loader=loader,
        batches=batches,
    )
    pruner.run(between_steps=between_steps)
    return pruner.model, pruner.stats()
