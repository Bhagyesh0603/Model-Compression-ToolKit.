"""Staged compression pipeline: prune -> recover -> distill -> quantize.

Every stage exports an ONNX artifact and measures it, so the result is a table
of real numbers per stage rather than one black-box output. That is what makes
a regression attributable: if latency got worse, you can see which stage did it.

Ordering is enforced, not just recommended. Quantization changes weight dtype,
which breaks structural pruning and blocks gradient-based fine-tuning, so it is
only ever allowed to run last.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

from .core import host as host_mod
from .core.benchmark import Measurement, compression_score, efficiency_score, measure
from .core.export import count_parameters, to_onnx
from .distill import DistillConfig, distill
from .pruning import PruneConfig, StructuredPruner, classifier_layers, recalibrate_bn
from .quantize import QuantConfig, QuantResult, quantize


@dataclass
class Stage:
    kind: str
    options: dict = field(default_factory=dict)


@dataclass
class PipelineResult:
    baseline: Measurement
    stages: list[Measurement]
    host: dict
    quantization: QuantResult | None = None
    prune_stats: dict | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def final(self) -> Measurement:
        return self.stages[-1] if self.stages else self.baseline

    @property
    def accuracy_drop(self) -> float:
        """Percentage points lost against the baseline."""
        return (self.baseline.accuracy - self.final.accuracy) * 100.0

    @property
    def speedup(self) -> float:
        return self.baseline.latency.median_ms / self.final.latency.median_ms

    @property
    def size_reduction(self) -> float:
        return self.baseline.size_mb / self.final.size_mb

    @property
    def efficiency_score(self) -> float:
        return efficiency_score(self.baseline, self.final)

    def table(self) -> str:
        rows = [self.baseline.summary()] + [m.summary() for m in self.stages]
        width = max(len(r) for r in rows)
        return "\n".join([rows[0], "-" * width] + rows[1:])

    def summary(self) -> str:
        lines = [
            self.table(),
            "",
            f"accuracy   {self.baseline.accuracy * 100:.2f}% -> {self.final.accuracy * 100:.2f}%  "
            f"({self.accuracy_drop:+.2f} pp)",
            f"latency    {self.baseline.latency.median_ms:.3f} -> {self.final.latency.median_ms:.3f} ms  "
            f"({self.speedup:.2f}x)",
            f"size       {self.baseline.size_mb:.2f} -> {self.final.size_mb:.2f} MB  "
            f"({self.size_reduction:.2f}x)",
            f"efficiency {self.efficiency_score:.4f}  (harmonic mean of accuracy and compression)",
        ]
        for warning in self.warnings:
            lines.append(f"! {warning}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "host": self.host,
            "baseline": self.baseline.as_dict(),
            "stages": [m.as_dict() for m in self.stages],
            "prune_stats": self.prune_stats,
            "quantization": (
                {
                    "precision": self.quantization.precision,
                    "coverage": self.quantization.coverage,
                    "quantized_ops": self.quantization.quantized_ops,
                    "unquantized_ops": self.quantization.unquantized_ops,
                }
                if self.quantization
                else None
            ),
            "totals": {
                "accuracy_drop_pp": self.accuracy_drop,
                "speedup": self.speedup,
                "size_reduction": self.size_reduction,
                "efficiency_score": self.efficiency_score,
                "compression_score": compression_score(self.baseline, self.final),
            },
            "warnings": self.warnings,
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=2))
        return path


class Pipeline:
    """Chainable stage builder. Nothing executes until `run()`.

    ``Pipeline(model, calib, test).prune(0.5).recover().distill(10).quantize()``
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader,
        test_loader,
        train_loader=None,
        input_shape: tuple[int, ...] = (1, 3, 32, 32),
        workdir: str | Path = "artifacts",
        warmup: int = 50,
        runs: int = 200,
        accuracy_limit: int | None = None,
    ):
        # The untouched original is both the accuracy reference and the
        # distillation teacher, so it must survive every in-place stage.
        self.original = copy.deepcopy(model).cpu().eval()
        self.model = copy.deepcopy(model).cpu().eval()
        self.calibration_loader = calibration_loader
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.input_shape = input_shape
        self.workdir = Path(workdir)
        self.warmup = warmup
        self.runs = runs
        self.accuracy_limit = accuracy_limit
        self.stages: list[Stage] = []
        self.workdir.mkdir(parents=True, exist_ok=True)

    def prune(self, ratio: float = 0.5, align: int = 16, steps: int = 1, criterion: str = "magnitude", **kwargs):
        self.stages.append(
            Stage("prune", {"ratio": ratio, "align": align, "steps": steps, "criterion": criterion, **kwargs})
        )
        return self

    def recover(self, batches: int | None = None):
        self.stages.append(Stage("recover", {"batches": batches}))
        return self

    def distill(self, epochs: int = 10, loss: str = "kd", **kwargs):
        self.stages.append(Stage("distill", {"epochs": epochs, "loss": loss, **kwargs}))
        return self

    def quantize(self, precision: str = "auto", **kwargs):
        self.stages.append(Stage("quantize", {"precision": precision, **kwargs}))
        return self

    def preflight(self) -> str:
        """What this host can do and what the plan implies — before spending compute."""
        host = host_mod.probe()
        calibration_size = sum(len(b[0]) for b in self.calibration_loader)
        protected = [type(m).__name__ for m in classifier_layers(self.model)]

        lines = [
            "PRE-FLIGHT",
            "=" * 60,
            host.describe(),
            "",
            f"model     : {count_parameters(self.model):,} parameters, input {tuple(self.input_shape)}",
            f"calibration: {calibration_size} samples (must come from the train split)",
            f"protected : classifier {protected} is never pruned",
            f"plan      : {' -> '.join(s.kind for s in self.stages) or '(no stages)'}",
        ]

        quant_stages = [i for i, s in enumerate(self.stages) if s.kind == "quantize"]
        if quant_stages and quant_stages[0] != len(self.stages) - 1:
            lines.append("! quantize is not last — it will be reordered or rejected at run()")
        return "\n".join(lines)

    def _measure(self, name: str, model: nn.Module, precision: str = "fp32", notes: str = "") -> Measurement:
        artifact = to_onnx(model, self.workdir / f"{name}.onnx", self.input_shape)
        return measure(
            stage=name,
            artifact=artifact,
            loader=self.test_loader,
            precision=precision,
            parameters=count_parameters(model),
            input_shape=self.input_shape,
            warmup=self.warmup,
            runs=self.runs,
            accuracy_limit=self.accuracy_limit,
            notes=notes,
        )

    def run(self, verbose: bool = True) -> PipelineResult:
        kinds = [stage.kind for stage in self.stages]
        if "quantize" in kinds and kinds.index("quantize") != len(kinds) - 1:
            raise ValueError(
                "quantize must be the last stage — it changes weight dtype, which breaks "
                "structural pruning and blocks further fine-tuning"
            )

        def log(message):
            if verbose:
                print(message, flush=True)

        results: list[Measurement] = []
        warnings: list[str] = []
        quant_result = None
        prune_stats = None
        example = torch.randn(*self.input_shape)

        baseline = self._measure("baseline", self.original)
        log(baseline.summary())

        for index, stage in enumerate(self.stages):
            name = f"{index + 1}_{stage.kind}"

            if stage.kind == "prune":
                pruner = StructuredPruner(self.model, example, PruneConfig(**stage.options))
                # BN statistics go stale the moment channels are removed, so
                # refresh them between iterative steps or the next importance
                # ranking is computed on a mis-normalised network.
                pruner.run(
                    between_steps=lambda m, _: recalibrate_bn(m, self.calibration_loader, batches=8)
                )
                self.model = pruner.model
                prune_stats = pruner.stats()
                log(f"  {prune_stats.summary()}")
                measurement = self._measure(name, self.model, notes=prune_stats.summary())

            elif stage.kind == "recover":
                recalibrate_bn(self.model, self.calibration_loader, batches=stage.options.get("batches"))
                measurement = self._measure(name, self.model, notes="BN statistics re-estimated")

            elif stage.kind == "distill":
                options = dict(stage.options)
                train_data = options.pop("loader", None) or self.train_loader
                if train_data is None:
                    raise ValueError(
                        "distillation needs training data — pass train_loader= to Pipeline() "
                        "or loader= to .distill()"
                    )
                config = DistillConfig(**options)
                distill(self.model, self.original, train_data, config)
                measurement = self._measure(
                    name, self.model, notes=f"{config.loss} distillation, {config.epochs} epochs"
                )

            elif stage.kind == "quantize":
                source = to_onnx(self.model, self.workdir / f"{name}_fp32.onnx", self.input_shape)
                quant_result = quantize(
                    source,
                    self.workdir / f"{name}.onnx",
                    self.calibration_loader,
                    QuantConfig(**stage.options),
                )
                warnings.extend(quant_result.warnings)
                log(f"  {quant_result.summary()}")
                measurement = measure(
                    stage=name,
                    artifact=quant_result.path,
                    loader=self.test_loader,
                    precision=quant_result.precision,
                    parameters=count_parameters(self.model),
                    input_shape=self.input_shape,
                    warmup=self.warmup,
                    runs=self.runs,
                    accuracy_limit=self.accuracy_limit,
                    notes="; ".join(quant_result.warnings),
                )

            else:
                raise ValueError(f"unknown stage {stage.kind!r}")

            results.append(measurement)
            log(measurement.summary())

        return PipelineResult(
            baseline=baseline,
            stages=results,
            host=host_mod.probe().as_dict(),
            quantization=quant_result,
            prune_stats=prune_stats.__dict__ if prune_stats else None,
            warnings=warnings,
        )
