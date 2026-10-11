"""Budget-driven compression — the decision layer.

You declare what you need ("latency<2ms, accuracy_drop<1%") and this finds the
*least* compression that gets there, escalating only when the cheaper option
falls short:

1. **Quantize only.** Cheapest, lowest risk, no retraining. Often enough.
2. **Add pruning**, binary-searching the ratio against the latency budget and
   using the sensitivity ranking to prune fragile layers less than robust ones.
3. **Add distillation**, but only if accuracy is what is failing — its entire
   job is recovering the accuracy the first two stages cost.

Stopping early is the point. A pipeline that always runs every stage spends
hours of fine-tuning to solve a problem quantization alone had already solved,
and gives back accuracy it never needed to lose.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .core import host as host_mod
from .core.benchmark import Measurement, efficiency_score, measure
from .pipeline import Pipeline, PipelineResult
from .pruning import LatencyGuidedReport, measure_latency_impact
from .sensitivity import SensitivityReport, analyze


def _merge_gentlest(overrides_a: dict[str, float], overrides_b: dict[str, float]) -> dict[str, float]:
    """Combine two ratio-override sources, keeping the gentler ratio wherever
    both name the same layer — protecting accuracy over maximizing speed
    whenever the two signals disagree, the same convention
    `_resolve_group_overrides` already uses for conflicting group overrides.
    """
    merged = dict(overrides_a)
    for name, ratio in overrides_b.items():
        merged[name] = min(merged.get(name, ratio), ratio)
    return merged


# A standalone decimal (a ratio, a millisecond figure) — but not the digits
# inside a layer path like "layer1.2.conv1", which identify *which* layer.
# Layers inside a nested nn.Sequential are named like "0.0" and "1.2", which
# look exactly like ratios; the prune warnings print those names quoted
# (['0.0'], {'0.0': 8}) or as a bare "0.0:" label, whereas a ratio in prose is
# never quoted or followed by a colon, so quotes and ':' mark a name to keep.
_STANDALONE_NUMBER = re.compile(r"""(?<![\w.'"])\d+\.\d+(?![\w.'":])""")


def _warning_key(warning: str) -> str:
    """What makes two warnings the same issue: identical text once ratio
    numbers are masked. Every prune trial in a search re-emits the same
    conflict at a different ratio, and printing each copy buries the one
    line that is actually new."""
    return _STANDALONE_NUMBER.sub("#", warning)


def _new_warnings(warnings: list[str], seen: set[str]) -> list[str]:
    """The warnings whose key hasn't been shown yet; records them as shown."""
    fresh = []
    for warning in warnings:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            fresh.append(warning)
    return fresh


_BUDGET_TERM = re.compile(
    r"\b(?P<key>latency|accuracy_drop|accuracy|size)\s*(?P<op>[<>]=?)\s*(?P<value>[\d.]+)\s*"
    r"(?P<unit>kb|mb|gb|ms|s|pp|%)?(?!\w)",
    re.IGNORECASE,
)


@dataclass
class Budget:
    """What the compressed model has to achieve. Unset fields are unconstrained."""

    max_latency_ms: float | None = None
    max_accuracy_drop_pp: float | None = None
    min_accuracy: float | None = None
    max_size_mb: float | None = None

    @classmethod
    def parse(cls, spec: str) -> Budget:
        """Parse 'latency<50ms, accuracy_drop<1%' into a Budget."""
        if not spec or not spec.strip():
            return cls()

        budget = cls()
        seen: dict[str, float] = {}
        matched = 0
        for term in _BUDGET_TERM.finditer(spec):
            matched += 1
            key = term.group("key").lower()
            operator = term.group("op")
            value = float(term.group("value"))
            unit = (term.group("unit") or "").lower()

            # Every term has one sensible direction: you cap latency, size and
            # accuracy loss, and you floor accuracy. Accepting the wrong operator
            # and silently treating it as the right one would invert the meaning
            # of the budget while still reporting "met".
            wants_upper_bound = key != "accuracy"
            if wants_upper_bound != operator.startswith("<"):
                expected = "<" if wants_upper_bound else ">"
                raise ValueError(
                    f"{key!r} takes {expected}, not {operator!r} — "
                    f"e.g. '{key}{expected}{value:g}{unit}'"
                )

            if key == "latency":
                field_name, resolved = "max_latency_ms", (value * 1000.0 if unit == "s" else value)
            elif key == "accuracy_drop":
                field_name, resolved = "max_accuracy_drop_pp", value
            elif key == "accuracy":
                field_name, resolved = "min_accuracy", (value / 100.0 if (unit == "%" or value > 1.0) else value)
            else:
                field_name = "max_size_mb"
                if unit == "kb":
                    resolved = value / 1024.0
                elif unit == "gb":
                    resolved = value * 1024.0
                else:
                    resolved = value

            # Two different values for the same key is an unresolvable
            # ambiguity — silently keeping whichever term matched last would
            # discard the other half of the caller's intent without a trace.
            # An exact repeat (e.g. composed from two code paths that agree)
            # is harmless and stays silent.
            previous = seen.get(key)
            if previous is not None and not math.isclose(previous, resolved, rel_tol=1e-9):
                raise ValueError(
                    f"{spec!r} specifies {key!r} twice with conflicting values "
                    f"({previous:g} and {resolved:g} after unit conversion) — remove one"
                )
            seen[key] = resolved
            setattr(budget, field_name, resolved)

        if not matched:
            raise ValueError(
                f"could not parse any budget term from {spec!r}; "
                "expected e.g. 'latency<50ms,accuracy_drop<1%'"
            )
        return budget

    @property
    def is_empty(self) -> bool:
        return not any(
            (self.max_latency_ms, self.max_accuracy_drop_pp, self.min_accuracy, self.max_size_mb)
        )

    def failures(self, baseline: Measurement, candidate: Measurement) -> list[str]:
        """Which constraints this candidate misses, phrased as the shortfall."""
        problems = []
        if self.max_latency_ms is not None and candidate.latency.median_ms > self.max_latency_ms:
            problems.append(
                f"latency {candidate.latency.median_ms:.3f} ms > {self.max_latency_ms:.3f} ms"
            )
        if self.max_accuracy_drop_pp is not None:
            drop = (baseline.accuracy - candidate.accuracy) * 100.0
            if drop > self.max_accuracy_drop_pp:
                problems.append(f"accuracy drop {drop:.2f} pp > {self.max_accuracy_drop_pp:.2f} pp")
        if self.min_accuracy is not None and candidate.accuracy < self.min_accuracy:
            problems.append(f"accuracy {candidate.accuracy * 100:.2f}% < {self.min_accuracy * 100:.2f}%")
        if self.max_size_mb is not None and candidate.size_mb > self.max_size_mb:
            problems.append(f"size {candidate.size_mb:.2f} MB > {self.max_size_mb:.2f} MB")
        return problems

    def met_by(self, baseline: Measurement, candidate: Measurement) -> bool:
        return not self.failures(baseline, candidate)

    def latency_failed(self, baseline: Measurement, candidate: Measurement) -> bool:
        return any(f.startswith(("latency", "size")) for f in self.failures(baseline, candidate))

    def accuracy_failed(self, baseline: Measurement, candidate: Measurement) -> bool:
        return any(f.startswith("accuracy") for f in self.failures(baseline, candidate))

    def describe(self) -> str:
        parts = []
        if self.max_latency_ms is not None:
            parts.append(f"latency <= {self.max_latency_ms:g} ms")
        if self.max_accuracy_drop_pp is not None:
            parts.append(f"accuracy drop <= {self.max_accuracy_drop_pp:g} pp")
        if self.min_accuracy is not None:
            parts.append(f"accuracy >= {self.min_accuracy * 100:g}%")
        if self.max_size_mb is not None:
            parts.append(f"size <= {self.max_size_mb:g} MB")
        return ", ".join(parts) or "unconstrained"


@dataclass
class Attempt:
    label: str
    plan: str
    result: PipelineResult
    failures: list[str]
    config: dict = field(default_factory=dict)

    @property
    def met(self) -> bool:
        return not self.failures

    @property
    def score(self) -> float:
        return efficiency_score(self.result.baseline, self.result.final)

    @property
    def prune_warnings(self) -> list[str]:
        """Pruning ran with `Pipeline(verbose=False)` during search, so these
        never reach the console unless surfaced here — and a warning here
        means the ratio this attempt reports (and any budget/efficiency
        conclusion drawn from it) may not reflect what was actually
        requested."""
        return (self.result.prune_stats or {}).get("warnings", [])


@dataclass
class SearchResult:
    budget: Budget
    attempts: list[Attempt]
    chosen: Attempt | None
    sensitivity: SensitivityReport | None = None
    host: dict = field(default_factory=lambda: host_mod.probe().as_dict())

    @property
    def met(self) -> bool:
        return self.chosen is not None and self.chosen.met

    def summary(self) -> str:
        lines = [f"budget: {self.budget.describe()}", ""]
        seen: set[str] = set()
        hidden = 0
        for attempt in self.attempts:
            mark = "MET " if attempt.met else "MISS"
            lines.append(f"[{mark}] {attempt.label:<28} {attempt.plan}")
            lines.append(f"        {attempt.result.final.summary()}")
            lines.append(f"        efficiency {attempt.score:.4f}")
            for failure in attempt.failures:
                lines.append(f"        - {failure}")
            fresh = _new_warnings(attempt.prune_warnings, seen)
            hidden += len(attempt.prune_warnings) - len(fresh)
            for warning in fresh:
                lines.append(f"        ! {warning}")
        lines.append("")
        if hidden:
            lines.append(
                f"({hidden} repeated pruning warning(s) hidden — the same issue at a different "
                "ratio; every attempt's full list is in Attempt.prune_warnings)"
            )
        if self.chosen is None:
            lines.append("no candidate produced — nothing ran")
        elif self.met:
            lines.append(f"CHOSEN: {self.chosen.label} — budget met, stopped here.")
        else:
            lines.append(
                f"BEST EFFORT: {self.chosen.label} — budget NOT met. "
                "Relax a constraint, or allow more distillation epochs."
            )
        return "\n".join(lines)


def _attempt(label: str, plan: str, build, budget: Budget, config: dict | None = None) -> Attempt:
    result = build()
    return Attempt(
        label=label,
        plan=plan,
        result=result,
        failures=budget.failures(result.baseline, result.final),
        config=config or {},
    )


def compress(
    model: nn.Module,
    calibration_loader,
    test_loader,
    budget: str | Budget = "",
    train_loader=None,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    workdir: str | Path = "artifacts/search",
    align: int = 16,
    distill_epochs: int = 10,
    distill_loss: str = "dkd",
    qat_epochs: int = 3,
    ratio_steps: int = 4,
    search_runs: int = 60,
    search_accuracy_limit: int | None = 2000,
    verbose: bool = True,
) -> SearchResult:
    """Find the least compression that meets `budget`. Escalates only as needed.

    During the search, accuracy is measured on a subset and latency on fewer runs
    to keep the loop affordable; the chosen configuration is then re-measured in
    full so the reported numbers are never the cheap estimates.
    """
    budget = Budget.parse(budget) if isinstance(budget, str) else budget
    workdir = Path(workdir)

    def log(message):
        if verbose:
            print(message, flush=True)

    def log_prune_warnings(candidate: Attempt) -> None:
        # Pipeline runs with verbose=False during search, so a pruning
        # attempt's own alignment/ratio-conflict warnings never print unless
        # repeated here — and they mean the ratio label on this attempt may
        # not be what actually got applied. Repeats of an already-shown
        # warning are suppressed here because the final summary counts them.
        for warning in _new_warnings(candidate.prune_warnings, shown_warnings):
            log(f"    ! {warning}")

    def pipeline(name: str, full: bool = False) -> Pipeline:
        return Pipeline(
            model,
            calibration_loader,
            test_loader,
            train_loader=train_loader,
            input_shape=input_shape,
            workdir=workdir / name,
            runs=200 if full else search_runs,
            warmup=50 if full else 20,
            accuracy_limit=None if full else search_accuracy_limit,
        )

    def run(pipe: Pipeline) -> PipelineResult:
        return pipe.run(verbose=False)

    def finalize(chosen: Attempt | None, report: SensitivityReport | None) -> SearchResult:
        if chosen is not None:
            _remeasure_full(chosen, budget, test_loader, input_shape)
            if verbose:
                print("\nfinal (full-fidelity re-measure):", flush=True)
                print(f"    {chosen.result.final.summary()}", flush=True)
        return SearchResult(budget=budget, attempts=attempts, chosen=chosen, sensitivity=report)

    log(pipeline("preflight").quantize().preflight())
    log("")
    log(f"budget: {budget.describe()}")

    attempts: list[Attempt] = []
    shown_warnings: set[str] = set()

    # 1. Quantization alone — no retraining, no architecture change.
    log("\n[1] quantize only")
    attempt = _attempt("quantize-only", "quantize", lambda: run(pipeline("q_only").quantize()), budget)
    attempts.append(attempt)
    log(f"    {attempt.result.final.summary()}")
    if attempt.met:
        log("    budget met — stopping.")
        return finalize(attempt, None)

    for failure in attempt.failures:
        log(f"    - {failure}")

    # 1.5. Quantize-only already hits latency/size but not accuracy — QAT
    #      recovers exactly that loss for less cost than full pruning plus
    #      distillation, so it's worth trying before escalating further.
    if train_loader is not None and budget.accuracy_failed(
        attempt.result.baseline, attempt.result.final
    ) and not budget.latency_failed(attempt.result.baseline, attempt.result.final):
        log(f"\n[1.5] QAT recovery ({qat_epochs} epochs)")
        qat_attempt = _attempt(
            "qat+quantize",
            f"qat({qat_epochs} epochs) -> quantize",
            lambda: run(pipeline("qat").qat(epochs=qat_epochs).quantize()),
            budget,
            config={"qat_epochs": qat_epochs},
        )
        attempts.append(qat_attempt)
        log(f"    {qat_attempt.result.final.summary()}")
        if qat_attempt.met:
            log("    budget met — stopping.")
            return finalize(qat_attempt, None)
        for failure in qat_attempt.failures:
            log(f"    - {failure}")

    # 2. Add pruning. Which layers to protect comes from measured sensitivity,
    #    not from a uniform ratio applied blindly across depths.
    log("\n[2] sensitivity analysis")
    report = analyze(model, calibration_loader, batches=4)
    log("    " + "\n    ".join(report.summary(top=5).splitlines()))

    best_pruned: Attempt | None = None
    hardest_pruned: Attempt | None = None
    latency_report: LatencyGuidedReport | None = None

    def merged_ratios(ratio: float) -> dict[str, float]:
        overrides = report.prune_ratios(ratio)
        if latency_report is not None:
            overrides = _merge_gentlest(overrides, latency_report.prune_ratios(ratio))
        return overrides

    # Only prune if speed or size is what's failing. If quantization already hit
    # the latency target and accuracy is the gap, pruning makes accuracy worse —
    # go straight to distillation instead.
    if budget.latency_failed(attempt.result.baseline, attempt.result.final):
        example = torch.randn(*input_shape)
        latency_report = measure_latency_impact(model, example, workdir=workdir / "latency_probe")
        log("    " + "\n    ".join(latency_report.summary(top=5).splitlines()))

        low, high = 0.0, 0.9
        for step in range(ratio_steps):
            ratio = round((low + high) / 2, 3)
            label = f"prune@{ratio:.3f}+quantize"
            log(f"\n[2.{step + 1}] prune ratio {ratio:.3f} (binary search)")
            candidate = _attempt(
                label,
                f"prune({ratio:.3f}, align={align}) -> recover -> quantize",
                lambda r=ratio: run(
                    pipeline(f"prune_{r:.3f}")
                    .prune(ratio=r, align=align, ratio_overrides=merged_ratios(r))
                    .recover()
                    .quantize()
                ),
                budget,
                config={"ratio": ratio},
            )
            attempts.append(candidate)
            log(f"    {candidate.result.final.summary()}")
            log_prune_warnings(candidate)

            if hardest_pruned is None or ratio > hardest_pruned.config["ratio"]:
                hardest_pruned = candidate

            hits_speed = not budget.latency_failed(candidate.result.baseline, candidate.result.final)
            if hits_speed:
                # Fast enough — remember it and try a gentler ratio.
                best_pruned = candidate
                high = ratio
                if candidate.met:
                    log("    budget met — stopping.")
                    return finalize(candidate, report)
                log("    fast enough, but accuracy is short — will try distillation.")
            else:
                low = ratio
                log("    still too slow — pruning harder.")

    # 3. Accuracy is what's failing, so recover it with distillation.
    #    Prefer the gentlest ratio that hit the speed target; if none did, keep
    #    the most aggressive one we tried rather than silently dropping pruning
    #    altogether — that one is the closest we got on latency.
    recovery_target = best_pruned or hardest_pruned or attempt
    if train_loader is not None and budget.accuracy_failed(
        recovery_target.result.baseline, recovery_target.result.final
    ):
        ratio = recovery_target.config.get("ratio", 0.0)
        log(f"\n[3] distillation recovery ({distill_loss}, {distill_epochs} epochs) at ratio {ratio:.3f}")

        def build_distilled():
            pipe = pipeline("distilled")
            if ratio > 0:
                pipe = pipe.prune(ratio=ratio, align=align, ratio_overrides=merged_ratios(ratio)).recover()
            return run(pipe.distill(epochs=distill_epochs, loss=distill_loss).quantize())

        distilled = _attempt(
            f"prune@{ratio:.3f}+distill+quantize" if ratio else "distill+quantize",
            f"prune({ratio:.3f}) -> recover -> distill({distill_loss}) -> quantize",
            build_distilled,
            budget,
            config={"ratio": ratio, "distill_epochs": distill_epochs, "loss": distill_loss},
        )
        attempts.append(distilled)
        log(f"    {distilled.result.final.summary()}")
        log_prune_warnings(distilled)
        if distilled.met:
            log("    budget met — stopping.")
            return finalize(distilled, report)

    # Nothing met the budget: return the best trade-off we actually measured.
    feasible = [a for a in attempts if a.met]
    chosen = max(feasible or attempts, key=lambda a: a.score) if attempts else None
    return finalize(chosen, report)


def _remeasure_full(attempt: Attempt, budget: Budget, test_loader, input_shape) -> None:
    """Re-measure the winner's artifacts at full fidelity, in place.

    The search deliberately measures cheaply — fewer timed runs, accuracy on a
    subset — because it runs many candidates. Reporting those estimates as the
    final result would understate the error bars, so the chosen artifact (which
    already exists on disk, no recompute needed) is measured properly and the
    budget verdict recomputed against the real numbers.
    """
    result = attempt.result

    def full(measurement: Measurement) -> Measurement:
        return measure(
            stage=measurement.stage,
            artifact=measurement.artifact,
            loader=test_loader,
            precision=measurement.precision,
            parameters=measurement.parameters,
            input_shape=input_shape,
            warmup=50,
            runs=200,
            accuracy_limit=None,
            notes=measurement.notes,
        )

    # Every row, not just the final one: the report shows per-stage deltas
    # against the baseline, so leaving intermediate stages at search fidelity
    # while the baseline is re-measured makes those columns compare numbers
    # taken under different conditions.
    result.baseline = full(result.baseline)
    result.stages = [full(stage) for stage in result.stages]
    attempt.failures = budget.failures(result.baseline, result.final)
