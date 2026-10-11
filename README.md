# compresskit

A CNN compression toolkit that optimises for **measured CPU latency**, not FLOPs.

Prune → recover → distill → quantize, with every stage exported to ONNX and
benchmarked on the machine you're actually running on.

---

## Why this exists

The compression algorithms are commoditised — Torch-Pruning, ONNX Runtime and
friends implement them well. What is missing is the layer that decides *how much*
of each to apply, and that tells you the truth about what you got:

- **Nothing checks the CPU before recommending INT8.** INT8 is only faster when
  the CPU has fused low-precision kernels (x86 VNNI/AMX, ARM DotProd/I8MM).
  Without them it is smaller and frequently *slower*. compresskit probes the host
  first and gates the precision on the answer.
- **FLOPs are not latency.** Two models with identical FLOPs routinely differ 2×
  in wall-clock. Every number here comes from a stopwatch on this host.
- **"Accuracy dropped" is not a diagnosis.** Per-layer sensitivity analysis names
  the specific layers responsible. The budget search automatically uses that
  ranking to prune fragile layers less than robust ones; the same ranking also
  picks which layers the search keeps in float32 when quantization alone misses
  only on accuracy (`keep_float=`), and you can do the same by hand with
  `report.exclude_nodes(...)` (see Tier 3).
- **Most models don't need the whole pipeline.** The budget search stops at the
  cheapest stage that meets your target instead of always running everything.

## Install

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e .
```

## Quickstart

### Tier 1 — declare a budget, let it decide

```python
import compresskit as ck
from compresskit.zoo.data import calibration_loader, test_loader, train_loader

result = ck.compress(
    model, calibration_loader(), test_loader(),
    budget="latency<0.5ms, accuracy_drop<1%",
    train_loader=train_loader(),
)
print(result.summary())
```

It tries quantization alone first. If only accuracy is short, it first keeps the
most sensitive layers in float32 (no training), then tries a brief QAT
(quantization-aware training) recovery pass before anything more
expensive. If latency or size is still short, it adds pruning — guided by
*two* measured signals, not FLOPs: per-layer sensitivity (which layers carry
the most signal) and per-group measured latency impact (which layers actually
cost speed when removed) — and only adds distillation if accuracy is still
the gap after that. It stops at the first configuration that fits.

### Tier 2 — drive the stages yourself

```python
result = (
    ck.Pipeline(model, calib, test, train_loader=train)
    .prune(ratio=0.5, align=16)   # SIMD-aligned channel counts
    .recover()                    # BN re-estimation, free, no backprop
    .distill(epochs=10, loss="dkd")
    .quantize(precision="auto")   # auto = gated on the host probe
    .run()
)
```

`.qat(epochs=3)` is also available as a stage — a cheaper alternative to
`.distill()` right before `.quantize()` when the only problem is quantization
rounding error, not general accuracy loss from pruning.

### Tier 3 — per-layer overrides, once the report names a layer

```python
report = ck.analyze(model, calib)
print(report.summary())

pipe.quantize(exclude_nodes=report.exclude_nodes(onnx_path, k=2))
pipe.prune(ratio=0.5, ratio_overrides=report.prune_ratios(0.5))
```

## CLI

```bash
compresskit probe                       # what this CPU can actually do
compresskit train --arch resnet20       # FP32 baseline on CIFAR-10
compresskit sensitivity --model ckpt.pt # rank layers by quantization damage
compresskit compress --model ckpt.pt --budget "latency<0.5ms,accuracy_drop<1%" --report out.html
compresskit pipeline --model ckpt.pt --prune 0.5 --distill 10 --report out.html
compresskit bench model.onnx            # time an existing artifact
```

## How it works

```
CPU probe ──────────────────────────────► gates the precision choice
     │
     ▼
prune ──────► BN recover ──────► distill ──────► QAT ──────► quantize
DepGraph      reset + recompute  teacher =       simulated    ORT static INT8/FP16
L1 magnitude  forward-only       the ORIGINAL    rounding,    QDQ, per-channel
or measured   no backprop        model           STE grad     always last
latency-guided     │                │                │              │
     │              │                │                │              │
     └──────────────┴────────────────┴────────────────┴──────────────┘
                          │
              measurement harness — every stage
              latency (median + p95, batch 1, 1 thread)
              accuracy (top-1 of the ONNX artifact)
              size (actual bytes on disk)
```

**QAT (quantization-aware training)** is the cheap first fallback when plain
quantization alone misses the accuracy budget: a short retrain with the
rounding error simulated in the forward pass (via a straight-through
estimator, since `.round()` has no gradient) so the weights compensate for it
before the final quantize. It only escalates to full pruning + distillation
if QAT alone isn't enough.

**Structured, not unstructured pruning.** Zeroing individual weights leaves the
tensor the same size; no maintained CPU runtime turns that sparsity into speed.
Removing whole channels shrinks the GEMM, which is where latency comes from.

**SIMD alignment is a hard constraint, not post-hoc rounding.** Channel counts
are held to multiples of 8/16/32 inside the pruner. Rounding afterwards silently
changes the ratio you asked for and leaves widths straddling vector-register
boundaries.

**The teacher is the original model.** No separate teacher to train or store —
the uncompressed model is already the best version of this exact function.

**Quantization is always last** and the pipeline enforces it. It changes weight
dtype, which breaks structural pruning and blocks gradient-based fine-tuning.

## The honesty rules

These are enforced in code, not just documented:

1. **Latency is measured, never estimated.** FP32, FP16 and INT8 are all timed
   through an identical ORT session config — same thread count, same batch, same
   warmup and run counts. Comparing precisions timed under different settings is
   the easiest way to publish a fake speedup.
2. **Calibration data comes from the train split.** BN re-estimation and
   quantization calibration both fit parameters to data. Using the test set there
   makes the reported accuracy meaningless.
3. **Accuracy is measured on the exported artifact**, not the torch module, so
   export bugs and quantization error show up in the number instead of hiding.
4. **Ops that silently stayed float32 are reported.** An op with no INT8 kernel
   is left alone by the quantizer — the file still shrinks, so a size-only report
   looks like a win while the speedup quietly fails to appear.
5. **FP16 on the CPU provider is labelled a size win, not a speed win**, because
   that is what it measures as.

## Project layout

```
compresskit/
  core/              cross-cutting infra every technique depends on
    host.py            CPU/ISA probe, INT8 gating, training device
    export.py          torch -> ONNX, and ONNX-node <-> torch-layer naming
    benchmark.py        the measurement harness (latency, accuracy, size, host context)
  pruning/            TECHNIQUE: structured channel pruning
    prune.py            Torch-Pruning DepGraph wrapper: L1 magnitude, SIMD alignment
    recover.py           batch-norm re-estimation
    latency_guide.py      per-group ranking by measured latency impact, not FLOPs
  quantization/        TECHNIQUE: INT8/FP16 quantization
    quantize.py          ORT static INT8 / FP16, op-coverage reporting
    qat.py                quantization-aware training fallback (STE fake-quant)
  distillation/         TECHNIQUE: knowledge distillation
    distill.py            KD and Decoupled-KD losses, teacher = original model
  sensitivity/          TECHNIQUE: per-layer sensitivity analysis
    analysis.py            per-layer one-at-a-time MSE / SQNR analysis
  pipeline.py         staged orchestration, ordering enforcement
  search.py           budget-driven search with early exit
  report.py           before/after tables, Pareto plot, sensitivity chart
  cli.py              command line interface
  zoo/                project-specific reference material, not the general API
    models.py           CIFAR ResNet-20/56, VGG-16-BN
    data.py             CIFAR-10 loaders with train / calibration / test role separation
    train.py            baseline training, checkpoint I/O
    har.py              human-action-recognition dataset/model support
experiments/
  train_baseline.py  train ResNet-56 / VGG-16-BN on CIFAR-10 (MPS)
  run_sweep.py       ratio x precision sweep -> CSV + Pareto report
  demo.ipynb         one image, original vs compressed, side by side
tests/
```

## Background

- `Research_Survey.md` — synthesis of the 13-paper literature survey behind the
  ordering and loss choices.
- `architecture.png` — the system architecture diagram.
- `archive/System_Design.md` — competitive analysis and the build-vs-wrap
  decisions.

## Tests

```bash
./.venv/bin/python -m pytest tests/ -q
```
