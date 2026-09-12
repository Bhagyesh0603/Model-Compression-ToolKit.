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


def accuracy_onnx(path: str | Path, loader, limit: int | None = None) -> float:
    """Top-1 of the exported artifact — the thing that actually ships.

    Measured on the ONNX file rather than the torch module so that export bugs
    and quantization error show up in the reported number instead of hiding.
    """
    session = make_session(path, threads=0)  # 0 = ORT picks; accuracy isn't timed
    name = session.get_inputs()[0].name

    correct = seen = 0
    for images, targets in loader:
        logits = session.run(None, {name: images.numpy().astype(np.float32)})[0]
        correct += int((logits.argmax(1) == targets.numpy()).sum())
        seen += len(targets)
        if limit is not None and seen >= limit:
            break
    return correct / max(seen, 1)


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

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["latency"] = self.latency.as_dict()
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
    """Accuracy + latency + size for one ONNX artifact, in one call."""
    artifact = Path(artifact)
    return Measurement(
        stage=stage,
        precision=precision,
        accuracy=accuracy_onnx(artifact, loader, limit=accuracy_limit),
        latency=benchmark(artifact, input_shape, warmup=warmup, runs=runs, threads=threads),
        size_mb=size_mb(artifact),
        parameters=parameters,
        artifact=str(artifact),
        notes=notes,
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
