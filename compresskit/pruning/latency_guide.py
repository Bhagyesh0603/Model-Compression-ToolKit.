"""Latency-lookup-table-guided pruning: measure real latency impact per group,
not FLOPs or sensitivity, to decide which layers are worth pruning hardest.

Sensitivity analysis (`sensitivity/analysis.py`) answers "which layer is
carrying the most accuracy signal." This answers a different question: "which
layer's channels actually cost latency when removed." A layer can be
accuracy-safe to prune yet contribute almost nothing to latency (little point
pruning it hard), while another contributes disproportionately (worth
pruning hardest). Both signals matter and are merged by `search.py`.

Cost note: one measurement here is a full export+benchmark cycle (deepcopy +
trace + build an ORT session + N timed runs), not a cheap forward pass, so
`warmup`/`runs` default much lower than the toolkit's normal benchmarking
settings -- this only needs to rank groups relative to each other, not report
a publishable number.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from ..core.benchmark import benchmark
from ..core.export import to_onnx
from .prune import PruneConfig, StructuredPruner, _dependency_groups, classifier_layers


@dataclass
class LayerLatencyImpact:
    """One prunable dependency group and what probe-pruning it actually cost."""

    members: list[str]
    latency_delta_ms: float

    @property
    def name(self) -> str:
        return self.members[0]


@dataclass
class LatencyGuidedReport:
    """Groups ranked by measured latency impact, most-worth-pruning first."""

    impacts: list[LayerLatencyImpact]
    probe_ratio: float

    def ranked(self) -> list[LayerLatencyImpact]:
        return sorted(self.impacts, key=lambda impact: impact.latency_delta_ms, reverse=True)

    def prune_ratios(self, base_ratio: float, boost: float = 1.2, dampen: float = 0.6, cap: float = 0.9) -> dict[str, float]:
        """Prune the high-latency-impact half harder, the low-impact half gentler.

        A group that barely moves latency when pruned isn't worth the accuracy
        risk of pruning it hard; a group that measurably speeds things up is
        exactly where the requested ratio should be spent.
        """
        ranked = self.ranked()
        if not ranked:
            return {}
        half = max(1, len(ranked) // 2)
        overrides: dict[str, float] = {}
        for index, impact in enumerate(ranked):
            ratio = min(base_ratio * boost, cap) if index < half else base_ratio * dampen
            for name in impact.members:
                overrides[name] = ratio
        return overrides

    def summary(self, top: int = 10) -> str:
        ranked = self.ranked()
        lines = [f"per-group measured latency impact (probe ratio {self.probe_ratio}, most-worth-pruning first):"]
        for impact in ranked[:top]:
            lines.append(f"  {impact.name:<34} delta {impact.latency_delta_ms:+.4f} ms  ({len(impact.members)} tied layer(s))")
        if len(ranked) > top:
            lines.append(f"  ... {len(ranked) - top} more")
        return "\n".join(lines)


def measure_latency_impact(
    model: nn.Module,
    example_inputs: torch.Tensor,
    workdir: str | Path | None = None,
    probe_ratio: float = 0.3,
    warmup: int = 10,
    runs: int = 30,
    align: int = 1,
) -> LatencyGuidedReport:
    """Benchmark the real latency effect of pruning each group in isolation.

    `align=1` here deliberately -- this is a probe to rank groups relative to
    each other, not a final pruning pass, so SIMD-alignment rounding (which
    can zero out a narrow group's effective ratio) would just add noise to a
    measurement that only needs to be directionally right.
    """
    ignored = classifier_layers(model)
    groups = _dependency_groups(model, example_inputs, ignored)

    def _scratch_dir():
        if workdir is not None:
            path = Path(workdir)
            path.mkdir(parents=True, exist_ok=True)
            return path, None
        tmp = tempfile.TemporaryDirectory()
        return Path(tmp.name), tmp

    scratch, tmp_handle = _scratch_dir()
    try:
        input_shape = tuple(example_inputs.shape)
        baseline_path = to_onnx(model, scratch / "baseline.onnx", input_shape)
        baseline_latency = benchmark(baseline_path, input_shape, warmup=warmup, runs=runs).median_ms

        impacts = []
        for index, members in enumerate(groups):
            overrides = {name: probe_ratio for name in members}
            config = PruneConfig(ratio=0.0, align=align, ratio_overrides=overrides)
            try:
                pruner = StructuredPruner(model, example_inputs, config, ignored_layers=ignored)
                pruner.step()
                candidate_path = to_onnx(pruner.model, scratch / f"probe_{index}.onnx", input_shape)
                candidate_latency = benchmark(candidate_path, input_shape, warmup=warmup, runs=runs).median_ms
            except (KeyError, RuntimeError):
                # A group that Torch-Pruning rejects at this probe ratio (e.g.
                # too narrow to remove even one channel) tells us nothing about
                # latency impact -- it's a hard zero, not a measurement error.
                candidate_latency = baseline_latency
            impacts.append(LayerLatencyImpact(members=members, latency_delta_ms=baseline_latency - candidate_latency))

        return LatencyGuidedReport(impacts=impacts, probe_ratio=probe_ratio)
    finally:
        if tmp_handle is not None:
            tmp_handle.cleanup()
