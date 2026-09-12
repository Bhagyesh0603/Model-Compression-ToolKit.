# Model Compression ToolKit — Literature Survey

Synthesis of 13 papers in `Research papers/`, covering pruning, quantization, and knowledge
distillation (KD) for compressing CNNs to run on resource-constrained hardware. Goal: extract a
concrete, defensible design for a **prune → distill → quantize** toolkit targeting low-end PCs.

---

## 1. The core ordering question: why prune → distill → quantize?

Two papers directly answer this ("Pipeline of Pruning, KD, and Quantization" and "PAPER 1")
and their pipelines agree on the order:

1. **Quantization must be last.** Converting weights to INT8/FP16 changes the underlying dtype,
   which breaks standard structural pruning and complicates further gradient-based fine-tuning.
   Once quantized, a model is much harder to prune or distill further (Chae & Seo's feasibility
   matrix).
2. **Distillation should bracket pruning, not just follow it once.** Chae & Seo's "Compression-
   Aware KD" (CAKD) reuses the *original uncompressed model* as the teacher — first to fine-tune
   the pruned model, then again during quantization-aware training (QAT). This avoids training/
   storing a separate large teacher, and beats a one-shot pruning-then-single-KD-pass design on
   the accuracy-vs-compression Pareto front.
3. **Iterating prune ↔ distill beats one-shot pruning.** Wang et al. ("Iterative KD and Pruning")
   show interleaving small pruning steps with KD fine-tuning every epoch (gradual, scheduled
   pruning rate) recovers substantially more accuracy than pruning once and distilling once —
   up to +11% over baselines at matched FLOPs.
4. **Structured/channel pruning, not unstructured.** Multiple papers (Software Model Compression
   Studio, He et al./AMC) note unstructured (magnitude) pruning via `torch.nn.utils.prune`
   produces sparse-but-not-smaller tensors — no real speedup on CPU without a sparse-aware
   kernel. For a **low-end PC / CPU target**, structured (filter/channel) pruning is required to
   get actual latency and memory wins.

**Recommended toolkit pipeline:**

```
Full-precision teacher (trained or pretrained)
        │
        ▼
Structured filter/channel pruning (magnitude, Taylor-sensitivity, or AMC-style RL-searched ratios)
        │  (iterative: small prune step → KD fine-tune → repeat, not one-shot)
        ▼
Compression-Aware KD fine-tune, teacher = ORIGINAL uncompressed model
        │
        ▼
Quantization-Aware Training (QAT), KD loss reused from original model's logits
        │
        ▼
INT8 / FP16 export → CPU inference (ONNX Runtime / oneDNN backend)
```

---

## 2. Per-technique findings

### Pruning
- **Importance scoring**: magnitude (simplest), Taylor-expansion loss-sensitivity (Wang et al.,
  better for iterative schedules), or RL-searched per-layer ratios (AMC/He et al. — a DDPG agent
  picks a continuous sparsity ratio per layer using an 11-feature state embedding, validated
  without fine-tuning as a fast reward proxy).
- **AMC's finding**: non-uniform per-layer pruning (e.g., prune 3×3 convs harder than 1×1 convs)
  beats hand-tuned uniform ratios — useful as a heuristic prior even without running full RL
  search ourselves.
- **Residual/skip-connection caution** (AEMC, LLM paper but architecturally transferable):
  pruning blocks in a ResNet-style network distorts the shortcut-vs-residual norm ratio at
  downstream layers ("amplitude mismatch"), which can cause disproportionate accuracy collapse.
  A cheap parameter-free per-layer rescaling calibrated from a handful of examples restores
  expected activation magnitudes without fine-tuning — worth adding as a post-prune calibration
  step for ResNet-family CNNs.
- **Schedule**: gradual pruning (small % per iteration, e.g., 2% steps over many iterations,
  PAPER 1) with fine-tuning between steps outperforms aggressive one-shot pruning.

### Quantization
- **Quantization-aware training (QAT) > post-training quantization (PTQ)** when accuracy matters:
  fake-quant nodes with straight-through estimators during training let the network adapt to
  quantization noise (Fang et al., QAT-FL) — worth adopting even outside the federated-learning
  context they studied.
- **Bit-width isn't "lower is always better."** Chen et al. found median bit-widths (6–8 bit)
  often outperform both very low (2–4 bit) and very high (9 bit) — and, notably, quantization can
  act as a *regularizer* that improves accuracy on large/overfitting-prone models. But this
  benefit shrinks or reverses on already-pruned/small models — don't assume quantization always
  helps once the network is already compact.
- **INT8 speed requires INT8-aware kernels.** Several papers flag that INT8 gives memory savings
  but no latency win without a runtime that actually has optimized INT8 kernels (e.g., ONNX
  Runtime, oneDNN) — critical for our CPU/low-end-PC target; depthwise-separable layers in
  particular often lack fast INT8 kernels (seen in the HAR paper, where INT8 cost 6–8% accuracy
  and wasn't worth it on their runtime).
- **Data-free quantization is viable.** VLCQ's asymmetric two-region scaling + Golomb-Rice
  variable-length coding needs no calibration dataset and reaches up to 12.25× compression at
  <4% accuracy loss — useful when a labeled calibration set isn't available in the toolkit's
  target workflow.
- **Clustering-based (K-means) quantization** is an alternative to fixed bit-width (Khan &
  Rafat): quantize weights to k learned centroids instead of a uniform grid; can be applied to
  the *teacher* as well as the student inside a distillation loop for extreme compression, though
  the paper's ~34,000× ratios showed a large train/test generalization gap on CIFAR-100 — a
  caution to always validate on held-out test data, not training accuracy.

### Knowledge Distillation
- **Standard loss**: `L = (1-α)·CE + α·T²·KD(softmax/T, teacher_softmax/T)`, α≈0.3, T≈4 recurs
  across papers (Chae & Seo, Software Model Compression Studio) — a safe default.
- **Reuse the original uncompressed model as teacher** (CAKD) rather than training/maintaining a
  separate teacher network — simpler pipeline, no extra teacher-training cost, and Chae & Seo
  show it beats conventional teacher-student KD on their Pareto plot.
- **Logit decoupling is a free upgrade.** Xu et al.'s ReKD/TISD-NTID separates "head" (similar-
  class) vs. "tail" (dissimilar-class) knowledge in the logits with tunable weights — pure loss-
  function change, no extra compute or architecture change, consistent +1–3% gains across
  datasets. Cheap to drop into our post-pruning/quantization fine-tune stage.
- **Middle-layer (feature) KD complements logit KD.** PAPER 1 combines a feature-map distance
  loss between teacher/student middle layers with logit KD plus noise-injected soft targets
  (Gaussian noise/dropout as regularization) — contributed to a 49.5× compression, >3.2×
  measured speedup on Raspberry Pi 3.
- **Distill to a smaller architecture, not just fewer weights.** The HAR paper's teacher is a
  cross-modal transformer; students are lightweight 1D-CNN / depthwise-separable CNN — 13–23×
  parameter reduction via architecture change + KD, before any quantization is even applied.

---

## 3. Cross-cutting cautions (things that can silently break the pipeline)

1. **Test-set validation at every stage** — extreme compression ratios can look excellent on
   training accuracy while hiding a large generalization gap (Khan & Rafat, CIFAR-100).
2. **Order-dependent side effects** — quantization changes dtype and can block further
   pruning/KD; always quantize last (Chae & Seo).
3. **Quantization's accuracy effect is not monotonic** — it can help (regularization on
   large/overfit models) or hurt (small/already-pruned models, depthwise-separable layers on
   some runtimes) — benchmark, don't assume.
4. **Hardware/runtime dependency** — INT8 memory savings ≠ INT8 latency savings without the
   right kernels; always benchmark on the actual target runtime (ONNX Runtime/oneDNN on CPU for
   our low-end-PC case), not just report theoretical FLOPs/size reduction.
5. **Structured vs. unstructured pruning** — only structured (channel/filter) pruning yields real
   speed/memory wins on CPU; unstructured sparsity needs specialized sparse kernels we likely
   won't have on a low-end PC.

---

## 4. Reference architecture / UX pattern

"Software_Model_Compression_Studio.pdf" is a designed-not-built product spec, useful as a UX/
architecture template rather than a technique source:
- Per-stage independent benchmarking: log accuracy, CPU/GPU latency (100+ warmup runs), and disk
  footprint after **each** of prune/quantize/distill, not just at the end.
- Pareto-frontier visualization (accuracy vs. size vs. latency) so users pick a trade-off point
  instead of getting one black-box output model.
- A job queue (their Redis+Celery design) to serialize compression jobs when GPU/CPU memory is
  constrained — directly relevant if our toolkit runs multiple compression jobs on a single
  low-end machine.
- Explicit unstructured-vs-structured pruning toggle, defaulting to structured for real footprint
  gains.

A recommended **Efficiency Score** metric from Chae & Seo, useful for ranking pipeline configs
instead of eyeballing accuracy or size alone:

```
CompressionScore = mean(FLOPs_drop_ratio, size_drop_ratio)
EfficiencyScore  = harmonic_mean(Accuracy, CompressionScore)
```

---

## 5. Paper-by-paper index

| # | File | Category | One-line takeaway |
|---|------|----------|--------------------|
| 1 | Communication_Efficient_Federated_Learning_With_QAT | Quantization (QAT) | Fake-quant during training + adaptive bit-width ramp beats PTQ |
| 2 | Composite HAR (KD + sensor fusion) | KD + quantization | Multimodal teacher → lightweight single-input CNN student, 13–23× smaller; INT8 hurt accuracy on their runtime |
| 3 | Engineering Reports — Khan (CQKD) | Prune + K-means quant + KD | Quantize the teacher too; watch train/test generalization gap at extreme ratios |
| 4 | Improving KD via Head/Tail Categories (ReKD) | KD (logit-based) | Free +1–3% accuracy via TISD/NTID logit decoupling, no extra compute |
| 5 | Iterative KD and Pruning (ITMC) | Prune + KD, iterative | Interleave small prune steps with KD every epoch, not one-shot |
| 6 | PAPER 1 (edge knowledge transfer) | Lightweight arch + KD + prune | GAP+separable convs, feature+logit KD, global magnitude pruning → 49.5× compression, Raspberry Pi benchmarked |
| 7 | Pipeline of Pruning, KD, Quantization (Chae & Seo) | Prune → CAKD → QAT | Defines the ordering rationale + reuse-original-model-as-teacher (CAKD) + Efficiency Score |
| 8 | Quantization of DNNs for Accurate Edge | Quantization | Quantization can regularize large models but effect vanishes on small/pruned ones; INT8 needs matching hardware kernels |
| 9 | Software Model Compression Studio | Tool/architecture spec | Reference UX: per-stage benchmarking, Pareto dashboard, job queue for constrained hardware |
| 10 | VLCQ | Quantization (PTQ, data-free) | Asymmetric 2-region scaling + Golomb-Rice coding, no calibration data needed, up to 12.25× |
| 11 | AMC (Yihui He et al., ECCV 2018) | Pruning (RL-automated) | DDPG agent searches per-layer pruning ratios under a FLOPs/latency budget; discovers non-uniform patterns |
| 12 | paper 2 (Dynamic Conv + Vision Mamba image compression) | Architecture (not classifier compression) | ODConv (cheap dynamic attention) + linear-complexity Mamba block; lightweight/heavyweight tiering pattern |
| 13 | paper 5 (AEMC, LLM depth pruning) | Pruning (structured, training-free) | "Amplitude mismatch" diagnosis for residual/skip connections after block pruning — transferable to ResNet-style CNN pruning |

---

## 6. Suggested next step

Build a minimal working prototype (ResNet or VGG-style CNN on CIFAR-10) implementing the
pipeline in §1, with the Efficiency Score (§4) as the acceptance metric, and per-stage
benchmarking (§4) as the toolkit's core UX — this validates the design against real numbers
before expanding to more architectures or datasets.
