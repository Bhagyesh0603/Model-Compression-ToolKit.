"""The measurement harness every stage reports into.

Two rules hold this together:

1. Latency is measured on this host, never estimated from FLOPs. A 4x smaller
   model that runs slower is a regression, and only a stopwatch catches that.
2. FP32, FP16 and INT8 are timed through an *identical* ORT session config
   (single thread, batch 1, same warmup/run counts). Comparing precisions timed
   under different settings is the easiest way to publish a fake speedup.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from . import host as host_mod
from .export import size_mb

WARMUP_RUNS = 50
TIMED_RUNS = 200
BENCH_THREADS = 1


def make_session(path: str | Path, threads: int = BENCH_THREADS) -> ort.InferenceSession:
    """The one session config used for every measurement of every precision."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


@dataclass
class Latency:
    median_ms: float
    p95_ms: float
    mean_ms: float
    stdev_ms: float
    runs: int
    warmup: int
    threads: int
    batch: int

    def as_dict(self) -> dict:
        return asdict(self)


def benchmark(
    path: str | Path,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = WARMUP_RUNS,
    runs: int = TIMED_RUNS,
    threads: int = BENCH_THREADS,
) -> Latency:
    """Time a single forward pass. Median is the headline; p95 shows the tail."""
    session = make_session(path, threads=threads)
    name = session.get_inputs()[0].name
    batch = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(warmup):
        session.run(None, {name: batch})

    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {name: batch})
        samples.append((time.perf_counter() - start) * 1000.0)

    samples.sort()
    return Latency(
        median_ms=statistics.median(samples),
        p95_ms=samples[min(int(len(samples) * 0.95), len(samples) - 1)],
        mean_ms=statistics.fmean(samples),
        stdev_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        runs=runs,
        warmup=warmup,
        threads=threads,
        batch=input_shape[0],
    )


@dataclass
class ClassMetrics:
    """Per-class precision/recall/F1, accumulated from one evaluation pass.

    Top-1 alone hides *where* a compressed model lost accuracy. Pruning capacity
    out of a network degrades confusable classes first, so a 2-point top-1 drop
    can be one class collapsing rather than every class slipping evenly — and
    only a per-class breakdown tells those apart.

    `confusion[t][p]` counts samples of true class t predicted as p.
    """

    confusion: list[list[int]]

    @property
    def num_classes(self) -> int:
        return len(self.confusion)

    @property
    def total(self) -> int:
        return sum(sum(row) for row in self.confusion)

    @property
    def support(self) -> list[int]:
        """How many test samples each class actually had."""
        return [sum(row) for row in self.confusion]

    @property
    def accuracy(self) -> float:
        total = self.total
        return sum(self.confusion[c][c] for c in range(self.num_classes)) / total if total else 0.0

    def precision(self) -> list[float]:
        scores = []
        for c in range(self.num_classes):
            predicted = sum(self.confusion[t][c] for t in range(self.num_classes))
            scores.append(self.confusion[c][c] / predicted if predicted else 0.0)
        return scores

    def recall(self) -> list[float]:
        return [
            self.confusion[c][c] / support if support else 0.0
            for c, support in enumerate(self.support)
        ]

    def f1(self) -> list[float]:
        return [
            2 * p * r / (p + r) if (p + r) else 0.0
            for p, r in zip(self.precision(), self.recall())
        ]

    @property
    def predicted(self) -> list[int]:
        """How many samples were predicted into each class (column sums)."""
        return [sum(self.confusion[t][c] for t in range(self.num_classes)) for c in range(self.num_classes)]

    @property
    def evaluated_classes(self) -> list[int]:
        """Classes with at least one test sample — the only ones macro recall
        can legitimately cover.

        `accuracy_limit` truncates the loader mid-stream during search, and the
        CIFAR test split is not shuffled by default, so a truncated pass can miss
        whole classes. Averaging a 0 in for a class that was never evaluated
        would report a macro recall far below what the model actually achieves.
        Recall is undefined with zero true samples, so this is the right gate
        for it — but precision only needs a prediction to be defined, see
        `_scored_classes` below.
        """
        return [c for c, support in enumerate(self.support) if support > 0]

    @property
    def _scored_classes(self) -> list[int]:
        """Classes with real signal on either axis — what macro precision and
        F1 should average over.

        A class need not have appeared as a true label to have a real,
        computable precision: if the model predicted it at all, `precision =
        TP / predicted` is defined even when `support == 0`, and a class that
        only ever received false positives has a genuinely damaging precision
        of 0.0 that must not be silently dropped from the average. Precision
        (and thus F1) is only truly undefined when a class was neither a true
        label nor ever predicted.
        """
        predicted = self.predicted
        return [c for c in range(self.num_classes) if self.support[c] > 0 or predicted[c] > 0]

    def _macro(self, scores: list[float], present: list[int]) -> float:
        return sum(scores[c] for c in present) / len(present) if present else 0.0

    @property
    def macro_precision(self) -> float:
        return self._macro(self.precision(), self._scored_classes)

    @property
    def macro_recall(self) -> float:
        return self._macro(self.recall(), self.evaluated_classes)

    @property
    def macro_f1(self) -> float:
        return self._macro(self.f1(), self._scored_classes)

    @property
    def partial(self) -> bool:
        """True when some class had no samples — macro figures cover a subset."""
        return len(self.evaluated_classes) < self.num_classes

    def as_dict(self) -> dict:
        return {
            "confusion": self.confusion,
            "support": self.support,
            "accuracy": self.accuracy,
            "precision": self.precision(),
            "recall": self.recall(),
            "f1": self.f1(),
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "evaluated_classes": self.evaluated_classes,
        }

    def summary(self, class_names: list[str] | None = None) -> str:
        precision, recall, f1, support = self.precision(), self.recall(), self.f1(), self.support
        lines = [f"{'class':<12}{'prec':>8}{'recall':>8}{'f1':>8}{'support':>9}"]
        for c in range(self.num_classes):
            label = class_names[c] if class_names and c < len(class_names) else str(c)
            lines.append(f"{label:<12}{precision[c]:>8.4f}{recall[c]:>8.4f}{f1[c]:>8.4f}{support[c]:>9,}")
        lines.append(
            f"{'macro':<12}{self.macro_precision:>8.4f}{self.macro_recall:>8.4f}{self.macro_f1:>8.4f}"
            f"{self.total:>9,}"
        )
        if self.partial:
            missing = sorted(set(range(self.num_classes)) - set(self.evaluated_classes))
            lines.append(f"! classes {missing} had no samples — macro averages cover the rest only")
        return "\n".join(lines)


def evaluate_onnx(path: str | Path, loader, limit: int | None = None) -> ClassMetrics:
    """Full confusion matrix for the exported artifact, in one pass.

    Measured on the ONNX file rather than the torch module so that export bugs
    and quantization error show up in the reported numbers instead of hiding.
    """
    session = make_session(path, threads=0)  # 0 = ORT picks; accuracy isn't timed
    name = session.get_inputs()[0].name

    matrix = None
    seen = 0
    for images, targets in loader:
        logits = session.run(None, {name: images.numpy().astype(np.float32)})[0]
        if matrix is None:
            classes = logits.shape[1]
            matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (targets.numpy().astype(np.int64), logits.argmax(1)), 1)
        seen += len(targets)
        if limit is not None and seen >= limit:
            break

    return ClassMetrics(confusion=matrix.tolist() if matrix is not None else [])


def accuracy_onnx(path: str | Path, loader, limit: int | None = None) -> float:
    """Top-1 of the exported artifact — the thing that actually ships."""
    return evaluate_onnx(path, loader, limit=limit).accuracy


@torch.no_grad()
def accuracy_torch(model: torch.nn.Module, loader, device: str = "cpu", limit: int | None = None) -> float:
    """Top-1 of a live torch module, for in-training checkpoints."""
    was_training = model.training
    model.eval().to(device)

    correct = seen = 0
    for images, targets in loader:
        logits = model(images.to(device))
        correct += int((logits.argmax(1).cpu() == targets).sum())
        seen += len(targets)
        if limit is not None and seen >= limit:
            break

    model.train(was_training)
    return correct / max(seen, 1)


@dataclass
class Measurement:
    """One row of the before/after table: a named stage, fully measured."""

    stage: str
    precision: str
    accuracy: float
    latency: Latency
    size_mb: float
    parameters: int
    artifact: str
    host: dict = field(default_factory=lambda: host_mod.probe().as_dict())
    notes: str = ""
    class_metrics: ClassMetrics | None = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["latency"] = self.latency.as_dict()
        payload["class_metrics"] = self.class_metrics.as_dict() if self.class_metrics else None
        return payload

    def summary(self) -> str:
        return (
            f"{self.stage:<22} {self.precision:<5} "
            f"acc {self.accuracy * 100:6.2f}%  "
            f"lat {self.latency.median_ms:7.3f} ms (p95 {self.latency.p95_ms:7.3f})  "
            f"size {self.size_mb:6.2f} MB  params {self.parameters:,}"
        )


def measure(
    stage: str,
    artifact: str | Path,
    loader,
    precision: str = "fp32",
    parameters: int = 0,
    input_shape: tuple[int, ...] = (1, 3, 32, 32),
    warmup: int = WARMUP_RUNS,
    runs: int = TIMED_RUNS,
    threads: int = BENCH_THREADS,
    accuracy_limit: int | None = None,
    notes: str = "",
) -> Measurement:
    """Accuracy + per-class metrics + latency + size for one ONNX artifact."""
    artifact = Path(artifact)
    # One evaluation pass yields both top-1 and the confusion matrix; computing
    # them separately would double the inference cost of every stage.
    metrics = evaluate_onnx(artifact, loader, limit=accuracy_limit)
    return Measurement(
        stage=stage,
        precision=precision,
        accuracy=metrics.accuracy,
        latency=benchmark(artifact, input_shape, warmup=warmup, runs=runs, threads=threads),
        size_mb=size_mb(artifact),
        parameters=parameters,
        artifact=str(artifact),
        notes=notes,
        class_metrics=metrics,
    )


def compression_score(baseline: Measurement, candidate: Measurement) -> float:
    """mean(size_drop, latency_drop) — the survey's CompressionScore."""
    size_drop = 1.0 - (candidate.size_mb / baseline.size_mb) if baseline.size_mb else 0.0
    speed_drop = 1.0 - (candidate.latency.median_ms / baseline.latency.median_ms) if baseline.latency.median_ms else 0.0
    return (max(size_drop, 0.0) + max(speed_drop, 0.0)) / 2.0


def efficiency_score(baseline: Measurement, candidate: Measurement) -> float:
    """harmonic_mean(accuracy, CompressionScore) — Chae & Seo's ranking metric.

    Harmonic mean is the point: a config that keeps accuracy but compresses
    nothing, or compresses hugely but destroys accuracy, both score near zero.
    """
    comp = compression_score(baseline, candidate)
    acc = candidate.accuracy
    if acc <= 0 or comp <= 0:
        return 0.0
    return 2 * acc * comp / (acc + comp)
