# Proposed System

## 1. Overview and Design Rationale

The literature reviewed in Section 2 (Related Work) converges on a shared pipeline shape — prune,
recover accuracy via distillation, then quantize last — but also exposes three recurring
evaluation weaknesses: reliance on FLOPs or parameter-count proxies in place of measured latency
[1], [2], [4], [6], [10]; headline results drawn from a single best-case configuration rather than
an average [2], [3], [6]; and, in the most extreme case, compression ratios validated against
training accuracy rather than held-out test accuracy [7]. The proposed system, *compresskit*, is
built around a single design principle that directly answers these gaps: every number the system
reports is either measured on the deployment target or explicitly labelled as a proxy, and the
system refuses to let one stage's shortcut become the next stage's silent assumption.

Architecturally, the system is a staged pipeline — *host probe → structured pruning → batch-norm
recovery → compression-aware distillation → post-training quantization* — wrapped in a budget-
driven search layer that decides how many of these stages a given model actually needs (Fig. 1,
`architecture.png`). Unlike the fixed five-stage pipelines proposed in [1] and [7], the system
treats quantization alone as the default first attempt and escalates to pruning and then
distillation only when a declared accuracy/latency/size budget is not yet met, since running every
stage unconditionally spends fine-tuning time recovering accuracy that a cheaper configuration
never would have lost.

## 2. Host Capability Probing

Both [5] and [6] independently observe that INT8's memory savings do not imply a latency win
without hardware support for low-precision matrix multiplication. The proposed system operationalizes
this observation directly, rather than treating it as a caveat to mention after the fact. Before
any compression stage runs, a host-probing module inspects the CPU's instruction set (via `numpy`'s
CPU-feature flags, supplemented by `sysctl` on Apple silicon and `/proc/cpuinfo` on Linux) for the
presence of fused low-precision dot-product instructions — AVX-512 VNNI, AVX-VNNI, AMX-INT8 on
x86, or DotProd/I8MM on ARM. The presence of any one of these features gates the system's default
precision recommendation: INT8 is proposed only when a fused kernel path exists; otherwise FP16 is
recommended and INT8 becomes an explicit, warned opt-in. This turns a static claim from the
literature ("INT8 needs matching hardware") into an executable precondition checked once per host
and attached to every downstream measurement (`HostContext`), so a benchmark result can never be
read without also knowing what hardware produced it.

## 3. Structured Pruning

Following the consensus in [1] and [5] that only structured (filter/channel) pruning yields real
latency and memory reduction on CPU — unstructured, magnitude-based sparsity leaves the underlying
GEMM the same size — the pruning module wraps Torch-Pruning's dependency-graph pruner
(`DependencyGraph`, `MetaPruner`) rather than applying per-weight masks. Channel importance is
scored by one of magnitude (L1/L2 group norm), a Taylor first-order loss-sensitivity criterion
matching the scoring approach used in [2], or a random baseline for ablation. Pruning proceeds in
scheduled steps with a recovery callback (batch-norm re-estimation, or a short distillation
fine-tune) run between steps — the gradual, interleaved schedule shown by [2] to outperform
one-shot pruning at a matched final ratio.

Two implementation problems are addressed that are not discussed in the reviewed literature but
follow directly from the residual-connection risk both [1] (ordering) and [4] (amplitude mismatch
in residual streams) raise in different forms:

- **Per-layer SIMD alignment.** Restricting kept-channel counts to multiples of 8/16/32 keeps
  channel widths on vector-register boundaries, which matters for realized CPU speedup. Applying
  one global alignment value, however, causes any layer whose width barely exceeds one alignment
  block to be silently left completely unpruned once Torch-Pruning's default flooring rejects the
  resulting group as invalid. The pruning module instead computes the nearest achievable aligned
  channel count *per layer*, only relaxing alignment on the specific layer that cannot support it,
  and records every relaxation or truly unprunable layer explicitly rather than allowing it to pass
  silently.
- **Residual dependency-group ratio resolution.** A residual (skip) connection ties every layer
  feeding the same addition into one Torch-Pruning dependency group that must share a single
  channel count. When sensitivity-driven per-layer protection (Section 5) assigns different ratios
  to two layers inside the same group, the underlying pruner silently keeps only one member's
  ratio. The module detects this conflict and resolves the whole group to the gentlest (most
  protective) ratio requested, logging the resolution. This is a different mechanism from the
  post-hoc, training-free amplitude rescaling proposed in [4] for LLM depth pruning — the present
  system prevents the shortcut/residual channel-count mismatch from being introduced in the first
  place, at the graph level, rather than correcting an amplitude mismatch after the fact — but it
  is aimed at the same underlying failure mode that pruning a residual architecture can silently
  corrupt a skip connection's expected shape or scale.

The classifier layer is always excluded from pruning, since its output width defines the class
count rather than the model's speed.

## 4. Batch-Norm Recovery

Removing channels invalidates a batch-norm layer's running statistics immediately, before any
importance ranking computed on the next pruning step (for iterative Taylor scoring) or any
distillation fine-tune can be trusted. A parameter-free recovery stage re-estimates batch-norm
running mean/variance with a forward-only pass (no backpropagation) over a handful of calibration
batches between pruning steps and after the final pruning step, at effectively no training cost.

## 5. Per-Layer Sensitivity Analysis

Rather than the reinforcement-learning search proposed by [3] to discover non-uniform per-layer
compression ratios, the system computes a direct, deterministic sensitivity ranking: each
quantizable layer's weights are fake-quantized (symmetric, per-channel, 8-bit round-trip) one layer
at a time, with every other layer left in floating point, and the resulting shift in the model's
output logits is measured against a floating-point reference over a few calibration batches (no
labels, no backpropagation) via mean-squared error and signal-to-quantization-noise ratio (SQNR).
Layers are ranked worst-SQNR-first. This ranking is deliberately reused for two decisions rather
than treated as quantization-only diagnostics: the most sensitive layers are passed to the
quantizer as float32 exclusions, and the same layers receive a scaled-down pruning ratio (half the
global ratio, by default) relative to less sensitive layers — the non-uniform, per-layer protection
that [3] obtains through an RL search is obtained here directly from a measured, cheap forward-pass
signal instead. Where a named exclusion has no corresponding layer in the exported ONNX graph, the
system raises rather than silently applying an empty exclusion list, since an empty exclusion list
is otherwise indistinguishable from "exclude nothing."

## 6. Compression-Aware Distillation

Following the reuse strategy validated in [1] (CAKD) — where the original, uncompressed model
serves as the teacher at both the post-pruning fine-tuning stage and the quantization-aware
training stage, avoiding the cost of training and storing a separate teacher network — the
distillation module always distills the pruned student against a frozen copy of the original,
uncompressed model. Two loss formulations are supported: a classical temperature-scaled
KL-divergence blend (`(1−α)·CE + α·T²·KL`, α≈0.3, T≈4, matching the default recurring across the
reviewed literature), and Decoupled Knowledge Distillation (DKD), which separates the KL term into
a target-vs-non-target component (TCKD) and a ranking-among-wrong-classes component (NCKD) with
independently tunable weights — addressing the same coupling problem that motivates the head/tail
logit decoupling in [10], via a different split (target-class vs. all others, rather than top-K vs.
tail-K). A cosine learning-rate schedule with linear warmup is used, since a freshly pruned network
starts from a perturbed optimum and a full-size first gradient step risks moving it somewhere later
epochs cannot recover from.

## 7. Post-Training Quantization

Quantization is always the final stage, enforced structurally by the pipeline rather than left as
a convention: attempting to schedule any stage after quantization raises an error, directly
operationalizing the dtype-incompatibility argument common to [1] and [7]. Static INT8
quantization is performed through ONNX Runtime in QDQ format with per-channel weight scales
(avoiding the resolution loss a single per-tensor scale imposes on every filter but the
largest-magnitude one) and calibration restricted to the training split, never the test split. An
op-coverage check parses the exported ONNX graph after quantization and reports, per operator type,
how many convolution/matmul nodes actually ended up wrapped in a quantized (QDQ) kernel path versus
left in floating point — directly surfacing the failure mode noted in [5] and [6], where a model
can shrink on disk while compute-heavy operators silently keep their original latency because no
INT8 kernel exists for them. A further correction addresses a residual-specific gap in ONNX
Runtime's own exclusion mechanism: excluding a convolution that feeds a residual `Add` from
quantization is otherwise silently ignored, because the quantizer decides whether to insert QDQ
pairs based on what the *consumer* node needs; the module detects this pattern and expands the
exclusion to the downstream `Add` automatically.

## 8. Budget-Driven Search

The search layer accepts a declarative budget (e.g., `"latency<0.5ms, accuracy_drop<1%"`) and
finds the least aggressive configuration that satisfies it, in three escalating attempts:
quantization alone; quantization with structured pruning added (binary-searched against the
latency constraint, using the sensitivity ranking from Section 5 to protect fragile layers); and,
only if accuracy — not latency — is still failing, distillation recovery added on top of the best
pruned candidate found. Each attempt is measured cheaply during search (fewer timed runs, accuracy
on a subset) to keep the search loop affordable, and the final chosen configuration is re-measured
at full fidelity before being reported, so that the numbers ultimately reported are never the
cheaper search-time estimates. Candidates are ranked, when none meets the full budget, by an
efficiency score defined as the harmonic mean of accuracy and a compression score (itself the mean
of the FLOPs-drop and size-drop ratios) — the same formulation proposed in [1] as a single metric
for comparing compression configurations without reducing the comparison to accuracy or size in
isolation.

## 9. Measurement Harness and Reporting Discipline

Every stage exports an ONNX artifact and is measured independently — latency (median and
95th-percentile, fixed batch size and thread count, with warmup runs discarded), accuracy (computed
on the exported artifact, not the original PyTorch module, so export or quantization errors surface
in the number rather than being hidden), and on-disk size — producing a per-stage table rather than
a single end-to-end number. This directly targets the proxy-metric substitution identified across
the reviewed literature (Section 2.6): FP32, FP16, and INT8 artifacts are timed through an
identical ONNX Runtime session configuration so that a reported speedup reflects the precision
change and not an incidental difference in measurement conditions; FP16 on the CPU execution
provider is explicitly reported as a size reduction rather than a speed improvement, since FP16
weights are cast back to float32 for computation on that provider and do not reduce latency.

## References

See Section 2 (Related Work) for full bibliographic entries [1]–[10].
