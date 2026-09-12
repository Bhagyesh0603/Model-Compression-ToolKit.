# Model Compression ToolKit — System Design

Synthesis of: the 13-paper literature survey (`Research_Survey.md`) + competitive research into
existing compression toolkits, CPU deployment reality, current SOTA algorithms, and UX/automation
patterns. This is the concrete "what to build" answer.

---

## 0. The thesis, in one paragraph

The pruning/quantization/distillation *algorithms* are commoditized — Torch-Pruning, ONNX Runtime,
Intel Neural Compressor, and AIMET already implement most of them well. **Nobody combines them
into one CPU-latency-aware pipeline that (a) knows what the target CPU can actually do before
recommending INT8, (b) tells you which layer is responsible when accuracy collapses, and (c)
decides how much to prune vs. quantize vs. distill against a latency/accuracy budget instead of
making you hand-tune three separate tools.** That combination is the product. We wrap the hard,
well-solved engineering problems (dependency-graph-safe pruning, INT8 kernels) and build the part
that's genuinely missing (the decision layer).

---

## 1. Why build this at all — the gap, evidenced

- **The two most complete existing frameworks are contracting, not expanding.** OpenVINO NNCF
  3.0.0 dropped Knowledge Distillation, Structural Pruning, and NAS entirely; TF-MOT hasn't shipped
  a release since Feb 2024; Microsoft NNI's repo was archived Sep 2024; Neural Magic's
  SparseML/DeepSparse (the only production CPU engine for unstructured sparsity) were deprecated
  and archived June 2025. There is no actively-developed library today that does prune+quantize+
  distill together and stays maintained.
- **No tool checks the target CPU before recommending INT8.** ONNX Runtime's own docs admit INT8
  speedup is a coin-flip-to-negative on pre-VNNI/AVX2-only hardware, yet no surveyed toolkit queries
  CPUID (VNNI/AVX2/AVX512/dotprod) before choosing a quantization scheme. Clean, ownable gap.
- **Per-layer accuracy-loss diagnosis is only half-solved.** Intel Neural Compressor's "Neural
  Insights" (per-op MSE tables) and AIMET's `QuantAnalyzer` are the state of the art, but neither is
  CPU-latency-aware, both are PTQ-only (not pruning), and neither is on-by-default.
- **Residual/skip-connection-safe structured pruning exists (Torch-Pruning's DepGraph) but lives in
  isolation** from quantization and distillation — nobody chains it into a full pipeline.
- **Accuracy-aware auto-tuning exists (Intel Neural Compressor) but is expensive**: full
  Bayesian/exhaustive search over configs, each trial re-running a full validation pass — impractical
  on a no-GPU low-end machine, which is exactly our user's hardware.

---

## 2. What we reuse vs. build fresh

| Piece | Decision | Why |
|---|---|---|
| Dependency-graph-aware structured pruning (residual/concat-safe channel removal) | **Wrap Torch-Pruning (DepGraph)** | Already solves the hard graph-tracing problem correctly; reimplementing is pure risk for no benefit. |
| Quantization numerics + CPU export | **Wrap ONNX Runtime** (`quantize_static`, QDQ, S8S8) + **PT2E** (`torchao.quantization.pt2e`) on the PyTorch side | Eager/FX quantization is being sunset in favor of PT2E; ORT is the only mature cross-platform CPU INT8 path. |
| PTQ accuracy-recovery primitives (bias correction, cross-layer equalization, AdaRound) | **Port/reuse AIMET's or INC's algorithms** | Well-tested, math-heavy, not our differentiator — re-deriving them is wasted effort. |
| Knowledge distillation training loop | **Build ourselves** | It's ~100 lines (teacher forward pass + CE/KL blend); every existing tool either dropped it (NNCF) or never had it (TF-MOT, Torch-Pruning) — this is exactly where we add value by integrating it, not by inventing new math. |
| CPU-ISA-aware capability detection & INT8/FP16 gating | **Build ourselves** | Nobody does this; it's the cleanest white-space gap found. |
| Per-layer sensitivity diagnosis tied to *measured* CPU latency (not just MSE/FLOPs) | **Build ourselves**, modeled on Neural Insights/AIMET's *approach* | Existing versions are PTQ-only and latency-blind; ours drives both diagnosis and the search loop. |
| Accuracy/latency-budget search | **Build ourselves** (greedy sensitivity-ranked + binary search, not Bayesian/RL) | Cheap enough to run on a CPU-only laptop; mirrors NNCF's proven greedy approach rather than INC's expensive strategy zoo. |
| Structural reparameterization (RepVGG/MobileOne-style train-multi-branch → fuse-to-single-branch) | **Build as an independent, opt-in track** | Needs zero special runtime support (pure algebraic fusion at export), doesn't interact with the rest of the pipeline, real measured CPU wins (MobileOne-S4: 26.6ms vs RepVGG-B1's 193.7ms). Low risk, good payoff. |

---

## 3. The pipeline — concrete algorithm choices

```
┌─────────────────────────────────────────────────────────────────┐
│ 0. CPU capability probe (once, cached)                          │
│    detect VNNI / AVX2 / AVX512 / ARM dotprod-NEON via cpuid      │
│    → gates which quantization path is even offered              │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│ 1. (optional, independent) Structural reparameterization         │
│    train/load with RepVGG/MobileOne-style multi-branch blocks    │
│    → algebraic fusion to single-branch conv before export        │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│ 2. Structured pruning (Torch-Pruning DepGraph engine)            │
│    - magnitude-based importance (default) / Taylor (aggressive)  │
│    - candidate channel counts constrained to multiples of        │
│      8/16/32 as a HARD constraint in the search, not post-hoc    │
│      rounding (avoids the SIMD "staircase" effect)               │
│    - static per-layer latency lookup table (benchmark each       │
│      candidate width once offline) instead of FLOPs as the       │
│      hardware-aware search signal                                │
│    - iterative: small step → short fine-tune → repeat            │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│ 3. Compression-Aware KD fine-tune                                │
│    teacher = the ORIGINAL uncompressed model (no separate        │
│    teacher to train/store)                                       │
│    loss = DKD (Decoupled KD) — logit-only, no feature hooks,     │
│    matches/beats feature-based methods with far less complexity  │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│ 4. Quantization — gated by the CPU probe from step 0             │
│    - VNNI/AVX512-VNNI/dotprod present → INT8 PTQ                 │
│      (AdaRound/BRECQ-style block reconstruction) as default;     │
│      escalate to QAT only if PTQ drop > threshold (e.g. >2pp)    │
│      or task is detection/segmentation                           │
│    - AVX2-only/no-VNNI → default to FP16, offer INT8 as          │
│      opt-in with a loud "may be slower, benchmark before use"    │
│    - export: PT2E → ONNX (opset ≥13) → ORT static quantization   │
│      (QDQ, S8S8, per-channel weights, reduce_range on AVX2)      │
│    - KD reapplied during QAT using the same original model's     │
│      logits (second CAKD pass)                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
              Final artifact: ONNX INT8/FP16, benchmarked on the
              ACTUAL target machine, not estimated from FLOPs
```

**Deliberately deferred / rejected:**
- Unstructured pruning as a *latency* feature — no maintained CPU runtime exploits it since
  DeepSparse's death (June 2025); offer it only as a disk-size feature, clearly labeled as such.
- Joint (single-optimization) prune+quantize+distill — literature is contested and unsettled on
  even the ordering question; not worth the complexity for v1. Revisit in v2 with our own
  ordering-ablation harness rather than trusting one paper's claimed-optimal order.
- Weight clustering/K-means quantization — optimizes disk footprint via a codebook lookup, not
  CPU throughput; doesn't serve our stated target.
- Low-rank/tensor decomposition — needs heavy retraining, composes poorly with the small 3x3/1x1
  kernels in modern efficient CNNs. v2 at best.
- NAS — off-mission; we compress models users bring, not design new architectures.

---

## 4. The decision layer (this is the actual product)

Not a fixed pipeline everyone runs in full — a budget-driven search that stops as soon as it's
good enough, per your earlier "don't over-apply techniques" instinct, now made concrete:

1. **Pre-flight readiness report** (loud by default, before any compute is spent): CPU ISA
   detected, ops with no fused INT8 kernel (will silently stay FP32 otherwise), residual/concat
   groups Torch-Pruning will constrain together, estimated calibration-set size needed.
2. **User declares a budget**: `compress(model, latency<50ms, accuracy_drop<1%)`.
3. **Try quantization alone first** (cheapest, lowest-risk per §3 step 4). Measure on real
   hardware. If budget met, stop.
4. **If not met, run per-layer sensitivity analysis** (one-at-a-time toggle, MSE + SQNR per
   layer, ~O(N) forward passes — cheap, no GPU needed) to find which layers are actually costing
   accuracy, and add structured pruning, using the sensitivity ranking to decide which layers to
   prune more/less aggressively. Measure again.
5. **If accuracy dropped too much from pruning/quantization, add the DKD distillation fine-tune**
   (§3 step 3) — its whole job is recovering exactly this loss. Measure again.
6. **Stop as soon as the budget is hit** — never runs a stage that isn't needed. Rank any
   remaining candidate configs by Efficiency Score = harmonic_mean(accuracy, mean(FLOPs_drop,
   size_drop)) from the original 13-paper survey.

This greedy sensitivity-ranked + binary-search approach (validated by NNCF's own
`quantize_with_accuracy_control`) replaces expensive Bayesian/RL search (Intel Neural Compressor's
strategy zoo, AMC's RL agent) — both need dozens-to-hundreds of full evaluation passes, unrealistic
on a CPU-only laptop.

---

## 5. API shape (progressive disclosure — targets the #1 documented complaint: config sprawl)

```python
# Tier 1 — one line, budget-driven, for most users
result = mctk.compress(model, calib_data, budget="latency<50ms,accuracy_drop<1%")

# Tier 2 — staged pipeline, for users who want stage control but not per-op config
pipeline = (
    mctk.Pipeline(model)
    .prune(criterion="magnitude", align_simd=16)
    .distill(loss="dkd")
    .quantize(prefer="int8")   # auto-falls back to fp16 if CPU lacks VNNI
)
result = pipeline.run(calib_data)

# Tier 3 — per-layer override, only surfaced when the sensitivity report names a specific layer
pipeline.quantize(overrides={"layer4.2.conv2": "fp16"})
```

Config files are generated/exportable YAML for reproducibility, never mandatory input.

---

## 6. Reporting

Per run, an HTML report (not a bespoke experiment tracker — emit MLflow-compatible metrics instead
of reinventing longitudinal tracking):
- Pareto scatter: accuracy vs. **measured** CPU latency vs. disk size (never FLOP-estimated).
- Layer sensitivity panel (MSE/SQNR per layer) driving both the search and the "why did accuracy
  drop" question.
- A model-relationship view showing which stage (prune/distill/quantize) produced which variant,
  so users can compare runs without juggling tabs (the documented pain point from HCI research on
  compression tool usage).

---

## 7. v1 / v2 / rejected scope

| Feature | v1 | v2 | Rejected |
|---|---|---|---|
| Torch-Pruning-wrapped structured pruning, SIMD-aligned | ✅ | | |
| Magnitude pruning criterion | ✅ | | |
| Taylor/Fisher pruning criterion | | ✅ (aggressive mode) | |
| Static per-layer latency lookup for hardware-aware pruning | ✅ | | |
| ONNX Runtime INT8 PTQ (AdaRound/BRECQ-style) | ✅ | | |
| CPU-ISA capability detection, INT8/FP16 auto-gating | ✅ | | |
| QAT fallback (accuracy-gated) | ✅ | | |
| DKD logit distillation, original-model-as-teacher (CAKD) | ✅ | | |
| Per-layer sensitivity diagnosis (MSE/SQNR) | ✅ | | |
| Greedy sensitivity-ranked + binary search auto-tuning | ✅ | | |
| RepVGG/MobileOne-style reparameterization | ✅ (independent track) | | |
| Pre-flight readiness report | ✅ | | |
| 3-tier progressive-disclosure API | ✅ | | |
| HTML Pareto + sensitivity report | ✅ | | |
| Feature-based KD (ReviewKD/CRD) | | ✅ (accuracy ceiling option) | |
| Joint (single-optimization) prune+quant+distill | | ✅ (pending our own ordering ablations) | |
| Low-rank/tensor decomposition | | ✅ (niche) | |
| Weight clustering / K-means quantization | | | ❌ optimizes disk not latency |
| Unstructured pruning as a *latency* feature | | | ❌ no maintained CPU runtime exploits it |
| NAS for new architectures | | | ❌ off-mission |
| Bayesian/RL config search | | | ❌ too expensive on CPU-only hardware |

---

## 8. Open questions / contested findings to validate ourselves, not trust blindly

- **Stage ordering isn't fully settled in the literature.** Most evidence (Chae & Seo's CAKD,
  "Prune-Quantize-Distill") supports prune → distill → quantize, but at least one 2026 paper argues
  "weaker perturbation first" (quantize → prune) for different model families. Recommendation:
  build a lightweight ordering-ablation harness in v1 that tests 3-4 permutations on our own
  benchmark models rather than hard-coding one paper's claimed-optimal order.
- **Some cited 2026 papers (e.g. arXiv:2604.xxxxx, arXiv:2603.xxxxx range) are very recent/single-
  paper evidence** — treat their specific numbers as directional, not gospel, until we reproduce
  them ourselves.
- **Joint optimization (GETA-style) vs. sequential** — unresolved, deferred to v2 pending our own
  measurement rather than adopting either side's claim.

---

## 9. Suggested build order

1. CPU capability probe + pre-flight readiness report (small, immediately useful, no ML risk).
2. Torch-Pruning wrapper with SIMD-aligned constraint + magnitude criterion.
3. DKD distillation loop with original-model-as-teacher.
4. ONNX Runtime PTQ export path with ISA-gated INT8/FP16 selection.
5. Per-layer sensitivity diagnosis + greedy budget search tying 2-4 together.
6. HTML reporting.
7. RepVGG/MobileOne reparameterization as an independent add-on.

Validate at every step on a real ResNet/VGG-on-CIFAR-10 prototype, benchmarked on actual low-end
hardware (or at minimum an AVX2-only VM) — not just FLOPs/param counts — since that's the single
most-repeated failure mode across every piece of research above.
