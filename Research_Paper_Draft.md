# A Latency-Verified Compression Toolkit for Convolutional Neural Networks: Sensitivity-Guided Structured Pruning, SIMD-Aligned Channel Selection, and Host-Gated Quantization

---

## Abstract

Model compression for convolutional neural networks is normally reported through proxy metrics —
floating-point operation counts, parameter counts, or on-disk file size. This paper argues that
those proxies are systematically misleading on commodity CPU targets, and presents *compresskit*,
a staged compression toolkit in which every reported quantity is measured on the deployment
runtime rather than inferred from model structure. The system composes four stages — structured
channel pruning, batch-normalisation re-estimation, compression-aware knowledge distillation, and
post-training quantization — under four mechanisms that, to our knowledge, are not combined in
prior work. First, an instruction-set probe executed before compilation gates the target precision:
8-bit integer arithmetic is selected only when the host exposes a fused low-precision dot-product
path (AVX-512 VNNI, AVX-VNNI, AMX-INT8, ARM DotProd or I8MM), and half precision — a size reduction
rather than a latency one on the CPU execution provider, and reported as such — is selected
otherwise, because integer quantization without such a kernel reduces file size while frequently
increasing latency; a companion operator-coverage audit then verifies, on the exported graph, that
each compute operator was in fact fused into a low-precision kernel rather than merely surrounded by
one. Second, channel counts are constrained to vector-register-aligned multiples
through a per-layer rounding rule that computes the nearest achievable aligned width for each
layer independently, rather than applying one global alignment that silently leaves narrow layers
entirely unpruned. Third, per-layer pruning ratios are unified across the dependency groups that
residual connections induce, resolving toward the most protective member rather than letting the
underlying pruner pick arbitrarily among tied layers and discard the rest. Fourth, a label-free
per-layer sensitivity analysis — replacing one layer's weights at a time with their 8-bit
quantize–dequantize round trip and measuring the signal-to-quantization-noise ratio induced at the
output logits — produces a single ranking that supplies both the quantization exclusion set and the
non-uniform per-layer pruning ratios, obtaining the non-uniform allocation that prior work derives
from reinforcement-learning search at the cost of *L* + 1 forward passes. A declarative budget search escalates from
quantization alone to pruning and then distillation only when a stated latency or accuracy
constraint remains unmet, and re-measures the selected configuration at full fidelity before
reporting. We describe the mathematical formulation of each stage and the experimental protocol,
and report the measured floating-point baseline together with the projected compression outcomes
the protocol is designed to verify.

**Keywords:** model compression, structured pruning, knowledge distillation, post-training
quantization, SIMD alignment, layer sensitivity analysis, edge inference, ONNX Runtime, latency
benchmarking

---

## I. Introduction

### A. Background

Convolutional neural networks have become the default solution for image classification, detection
and segmentation, but the architectures that achieve competitive accuracy carry parameter counts
and arithmetic costs that assume a datacentre accelerator. Deploying these networks on
general-purpose CPUs — the processors actually present in laptops, industrial controllers, kiosks
and low-cost single-board computers — requires reducing both the memory footprint of the weights
and the wall-clock cost of a forward pass. Three families of techniques dominate the literature.
Pruning removes structural elements, either individual weights (unstructured) or entire channels
and filters (structured). Quantization reduces the numerical precision of weights and activations,
typically from 32-bit floating point to 8-bit integer or 16-bit floating point. Knowledge
distillation trains a compact student network against the output distribution of a larger teacher,
transferring the inter-class structure encoded in the teacher's soft predictions rather than only
the hard labels.

Each family was originally studied in isolation, and each has a mature body of results. The more
recent question — and the one that matters for anyone actually shipping a compressed model — is
how the three interact when composed into a single pipeline, and whether the composition's
reported gains survive contact with real hardware.

### B. Motivation

The motivation for this work arose from a specific and repeatable discrepancy. A network that has
been quantized to 8-bit integer precision is reliably about four times smaller on disk. It is not
reliably faster. Integer matrix multiplication outperforms floating-point matrix multiplication
only when the processor provides a fused low-precision dot-product instruction; on a CPU without
one, the runtime must widen the integers, perform the multiplication in a wider type, and requantize
the result, which can leave the integer path slower than the floating-point path it replaced. A
compression report that lists only file size records this outcome as an unambiguous success.

The same structural problem recurs elsewhere. Unstructured pruning drives a large fraction of
weights to zero without changing the dimensions of any tensor, so the general matrix multiply
executes at its original cost unless a sparse kernel is available. A floating-point operation count
that has been halved does not imply a halved execution time, because operation counts do not model
memory-access patterns, cache behaviour, or vector-register utilisation. In each case the proxy and
the objective diverge, and the direction of the divergence flatters the compression method.

### C. Problem Statement

The problem this work addresses is therefore not the design of a new compression *algorithm*. The
underlying algorithms — dependency-graph structured pruning, temperature-scaled distillation,
static integer quantization — are mature and well implemented in existing libraries. The problem is
the absence of a decision and verification layer: a system that determines *how much* of each
technique a particular model on a particular host requires, that refuses to apply a transformation
whose benefit the host cannot realise, and that reports what was actually achieved rather than what
the model's structure implies should have been achieved.

Concretely, four failure modes must be eliminated. A compression stage must not be able to report a
size reduction as a speed improvement. A pruning ratio requested by the user must not be silently
ignored for a subset of layers. An accuracy figure must not be computed on a model artifact
different from the one that will be deployed. And a per-layer decision — which layers to compress
hard, which to protect — must be derived from measurement rather than from a uniform default
applied blindly across network depth.

### D. Objectives

The objectives of this work are as follows:

1. To design a staged compression pipeline in which stage ordering is enforced structurally, so
   that a transformation which invalidates a later stage cannot precede it.
2. To formulate a per-layer SIMD-alignment rule that keeps pruned channel widths on
   vector-register boundaries without allowing a single narrow layer to force the relaxation of
   alignment across the entire network.
3. To derive a label-free, backpropagation-free per-layer sensitivity measure capable of ranking
   layers by the output distortion their compression induces, and to use that single ranking for
   two distinct decisions — quantization exclusion and non-uniform pruning allocation.
4. To gate the target numerical precision on a runtime probe of the host instruction set rather
   than on an assumption about the deployment target.
5. To define a budget-driven search that terminates at the least aggressive configuration
   satisfying a declared latency, accuracy and size constraint, and to specify a measurement
   protocol under which the reported figures are directly comparable across precisions.

### E. Research Contribution

This paper makes four contributions. The first is the *per-layer SIMD-aligned ratio resolution*
rule (Section III-E), which computes the nearest achievable aligned channel width independently per
layer and reduces alignment only for the specific layer whose width cannot accommodate it,
preserving full alignment elsewhere. The second is *dependency-group ratio unification*
(Section III-F), which resolves the conflict that arises when a residual connection ties layers
assigned different pruning ratios into a single group that must share one channel dimension,
resolving to the most protective ratio rather than permitting an arbitrary choice. The third is the
*dual use of a single SQNR-based sensitivity ranking* (Section III-G) to supply both quantization
exclusions and pruning ratios, replacing reinforcement-learning search with a deterministic
measurement whose cost is linear in layer count; the pruning-ratio half of this ranking is consumed
automatically by the budget search of Section III-J, while the exclusion half is exposed as an API
the caller wires explicitly, a gap discussed in Sections III-G and IV-E. The fourth is the *host-gated precision policy and
operator-coverage audit* (Section III-I), which together make it structurally impossible for the
system to report a file-size reduction as a latency improvement.

---

## II. Literature Survey

### A. Comparative Analysis

Table I summarises ten works spanning pipeline integration, structured pruning, quantization and
distillation, characterised along the dimensions that determine whether a reported gain transfers
to a CPU deployment target. The final column records a methodological limitation identified by
direct examination of each paper's own reported tables.

**TABLE I. COMPARATIVE ANALYSIS OF SURVEYED COMPRESSION LITERATURE**

| Ref. | Technique | Evaluation | Latency measured? | Identified limitation |
|------|-----------|-----------|-------------------|----------------------|
| [1] | Prune → CAKD → QAT → INT8 | ResNet-56/CIFAR-100, MobileNetV2/Tiny-ImageNet | No — FLOPs/size only; GPU used for training | Two FLOPs tables for the same configuration are mutually inconsistent |
| [2] | Iterative prune ↔ KD (ITMC) | Office-31, ImageCLEF-DA, PACS | No — FLOPs only; GPU training-only | Headline +11.2 pp is best task vs. weakest baseline; average gain is +1.3 pp |
| [3] | RL-searched per-layer ratios (AMC) | ImageNet, CIFAR-10, VOC 2007 | Yes — Titan Xp and Pixel 1 | Abstract claims +2.7% at 4×; table reports 5× with a 0.3-point margin |
| [4] | Training-free depth pruning (AEMC) | LLaMA-3.1-8B, Baichuan2, GPT-J | No — memory only | Aggregate gain masks per-task collapse (73.8% → 24.5% on one benchmark) |
| [5] | Bit-width sweep, 2–9 bit | MICCAI gland, ImageNet, TIMIT | No — explicitly states no CPU/GPU speedup | Median-bit-width claim contradicted by its own speech results (9-bit best) |
| [6] | Data-free PTQ + Golomb-Rice (VLCQ) | ImageNet, VOC | No — decoder throughput asserted, not measured | 12.25× headline holds for one configuration only; general claim is ~6× |
| [7] | Prune + K-means quant + KD (CQKD) | CIFAR-10/100, ResNet-18 | No | Distillation loss never implemented; abstract reports *training* accuracy (97.9% vs. 87.4% test) |
| [8] | Federated QAT | MNIST, CIFAR-10, FEMNIST | Partial — one inference figure | PTQ and QAT convergence bounds are algebraically identical |
| [9] | Lightweight arch + feature/logit KD + pruning | CIFAR-10, VGG-derived | Yes — GPU, Raspberry Pi 3, Cambricon | Stated 49.5× is 48.4× by its own parameter counts; teacher cannot run on the Pi at all |
| [10] | Logit decoupling by rank (ReKD) | CIFAR-100, Tiny-ImageNet, ImageNet, COCO | No | "No additional hyperparameters" contradicted by its own K ablation |

Two patterns are immediately visible. Only two of ten works [3], [9] report measured on-device
latency under a stated protocol — a third, [8], reports a single inference figure without one — and
both of the two are the works that also report the most modest and most carefully qualified gains.
Every remaining work motivates itself with deployment efficiency while reporting exclusively
structural proxies. This is the gap the present system is designed to close.

### B. Pipeline Composition and Stage Ordering

Chae and Seo [1] address the composition question directly rather than studying any single
technique. Their central structural argument is that quantization must be applied last: converting
weights to a low-precision integer representation changes the underlying datatype, which both
breaks structural pruning — the channel-removal operation assumes floating-point tensors it can
re-index — and blocks further gradient-based fine-tuning, since gradients cannot flow through the
quantized representation without additional machinery. Their pipeline therefore runs pruning,
distillation-assisted fine-tuning, quantization-aware training, and integer export in fixed order.
Their principal algorithmic contribution, Compression-Aware Knowledge Distillation, reuses the
original uncompressed network as the teacher at two separate points rather than training and
storing a separate large teacher. Evaluated on ResNet-56/CIFAR-100 and MobileNetV2/Tiny-ImageNet,
the method reports a monotonic top-1 decline from 71.84% at baseline to 63.70% at a pruning ratio
of 0.5. All reported results are structural: the GPU named in the paper is used for training only,
and the two floating-point operation tables given for the same configuration do not agree with each
other. The ordering argument is nonetheless sound and is adopted structurally in the present work.

Wang et al. [2] examine a narrower question — whether pruning and distillation should each be
applied once, or interleaved. Their method alternates distillation and adaptive channel pruning
every epoch, ranking channels by a first-order Taylor loss-sensitivity criterion and removing them
in small increments with fine-tuning immediately following each removal, inside a scheduled window
whose pruning rate ramps up and then decays. The interleaved schedule is reported to outperform
one-shot pruning at matched operation counts by margins as large as 11.2 and 18.6 percentage
points. Examination of the paper's aggregate tables shows these to be best-single-task comparisons
against the weakest available baseline; the average improvement is 1.3 points over the uncompressed
model and 4.4 points over the strongest competing method. As with [1], no latency is measured
despite explicit motivation by real-time and resource-constrained deployment.

### C. Structured Pruning and Per-Layer Allocation

He et al. [3] established that per-layer pruning ratios should not be uniform. Their AMC framework
trains a deep deterministic policy gradient agent that processes layers sequentially, encoding each
as an eleven-dimensional state — layer index, channel counts, spatial dimensions, stride, kernel
size, remaining operation budget, and the previous layer's action — and emitting a continuous
sparsity ratio, evaluated without fine-tuning as a fast reward proxy. On ImageNet the method
compresses ResNet-50 by 3.4–5× at 76.11% top-1 against a 76.13% baseline, and prunes MobileNet-V1
to 285 MFLOPs at 70.5% top-1, exceeding a width-scaled baseline that requires 325 MFLOPs for 68.4%.
Critically, AMC reports measured speedups of 1.53× on GPU and 1.95× on a mobile device. The
finding that non-uniform allocation outperforms uniform ratios is well supported; the abstract's
claim of a 2.7% margin at 4× reduction does not reconcile with a results table reporting 5%
reduction at a 0.3-point margin. The present work adopts AMC's conclusion — non-uniform per-layer
allocation is necessary — while rejecting its mechanism, since a reinforcement-learning search is
disproportionate to the problem when a direct measurement suffices.

Zhou et al. [4] identify a failure mode specific to residual architectures. Removing contiguous
blocks disturbs the trained ratio between the shortcut path and the residual path, because the
residual stream's direction remains stable with depth while its norm grows monotonically; later
blocks are calibrated to receive a particular norm, and supplying the wrong one causes destructive
overwriting rather than graceful degradation. Their correction is a parameter-free scalar rescaling
per pruned block, calibrated from three to four examples on the dense model with no gradient
updates. At 24.4% pruning of LLaMA-3.1-8B the method recovers average accuracy from 34.0% to 56.3%
against a 61.7% dense ceiling. Although evaluated on language models, the diagnosis is a property
of residual connections generally and transfers directly to residual CNNs. The limitation is that
the aggregate recovery conceals per-task collapse: on one benchmark accuracy falls from 73.8% to
24.5%, barely above the baselines the method is framed as correcting.

### D. Quantization and Hardware Dependence

Chen et al. [5] sweep bit-widths from 2 to 9 across segmentation, classification and speech tasks,
arguing that quantization regularises over-parameterised models such that a mid-range width
outperforms both extremes. The claim holds for their segmentation results, where accuracy peaks at
86.51–86.66% at 6–7 bits against 85.26–86.12% at 9 bits, but is contradicted by their own speech
results, where 9 bits is optimal outright. Of direct relevance here, the paper reports memory
reductions of 3.5–6.4× while explicitly stating that no speedup occurs on unmodified CPU or GPU
hardware, citing separate prior work for FPGA-only acceleration. This is the clearest statement in
the surveyed literature of the premise underlying the present system's precision gating.

Abdel-Salam et al. [6] propose a data-free post-training method that splits each weight tensor at
zero and quantizes the two regions with independent symmetric scales, then compresses the resulting
integers with a modified Golomb-Rice variable-length code. The reported 12.25× compression within a
4% accuracy loss corresponds to one specific configuration — per-channel quantization of ResNet-18
at 16 levels, yielding a 3.89-point drop from a 69.76% baseline; the paper's general claim across
models is a more modest 6×. The authors acknowledge that their scale-search heuristics are static
rather than adaptive, and the decoder's throughput is asserted rather than measured.

Khan and Rafat [7] combine L1-norm filter pruning with layer-wise K-means quantization applied to
both teacher and student, reporting compression ratios exceeding 34,000×. Two issues limit the
result. The composite distillation loss presented as a core contribution is described only as a
hypothetical formulation and is never exercised; the reported students are trained with plain
cross-entropy. Separately, the headline accuracies of 97.9% and 91.2% are training accuracies,
while the corresponding test accuracies in the paper's own table are 87.39% and 53.61% — gaps of
roughly 10 and 38 points. This is the most direct illustration in the surveyed literature of why
compression claims must be validated on held-out data at every stage.

Fang et al. [8] insert fake-quantization operators with straight-through gradient estimation during
local training in a federated setting, reporting 2–9 point improvements over post-training
quantization on MNIST and 2–4 points on CIFAR-10. The convergence bounds derived for the
quantization-aware and post-training variants are algebraically identical, and the claimed
advantage rests on an asymmetric relaxation applied to one term in an appendix derivation; the sole
heterogeneous benchmark is evaluated only through qualitative curves.

### E. Knowledge Distillation

Chen et al. [9] combine architectural lightweighting — global average pooling replacing fully
connected layers, separable convolutions replacing standard ones — with a two-part distillation
loss comprising a middle-layer feature-distance term and a logit term whose teacher targets are
perturbed with Gaussian noise as regularisation, followed by iterative magnitude pruning at 2% per
step across 15 steps. On CIFAR-10 the method reaches 75.96% against 74.54% for standard
distillation and 71.38% for a student trained from scratch. The work is among the few to benchmark
on edge hardware, but two headline figures do not survive checking: the stated 49.5× compression is
48.4× by the paper's own parameter counts, and the speedup attributed to a Raspberry Pi 3 cannot be
computed on that device, since the paper states the teacher exceeds its memory and cannot run.

Xu et al. [10] decouple the distillation objective by rank, partitioning the teacher's logits into
top-K and remaining categories and forming a separate divergence term over each with independent
weights. Gains of roughly 0.6 to 3.7 points over classical distillation are reported across
CIFAR-100, Tiny-ImageNet, ImageNet-1K and COCO. The stated absence of additional hyperparameters
and computational overhead is contradicted by the method's own three new tunable parameters and by
an ablation showing the optimal K to be dataset-dependent; no runtime measurement is reported
anywhere in the paper.

### F. Synthesis

Three gaps recur with enough regularity to be treated as structural rather than incidental. Proxy
metrics substitute for measurement in eight of the ten surveyed works. Headline figures are drawn
from best-case configurations or single favourable comparisons in at least three [2], [3], [6].
Validation against held-out data fails in the most aggressive compression regime [7]. These
observations define the design requirements adopted in Section III: measure on the target runtime,
report distributions rather than best cases, and evaluate every stage on the exported artifact
against a held-out split.

---

## III. Materials and Methods

### A. Dataset

Primary evaluation uses CIFAR-10: 60,000 colour images of 32×32 pixels across ten mutually
exclusive classes, partitioned into 50,000 training and 10,000 test images. The choice is
deliberate. The dataset is small enough that a full compression sweep — multiple pruning ratios
crossed with multiple precisions, each requiring export, calibration and timed benchmarking — is
tractable on a single machine, and it is standard enough in the compression literature [1], [7],
[9] that baseline accuracies are directly comparable.

Critically, the 50,000 training images are partitioned into three *disjoint functional roles* which
the system enforces separately:

- **Training split** — gradient updates during baseline training and distillation fine-tuning.
- **Calibration split** — a 512-image subset used for batch-normalisation re-estimation, for
  quantization activation-range calibration, and for sensitivity analysis. Drawn exclusively from
  the training partition.
- **Test split** — the untouched 10,000 test images, used only for reporting.

The separation is load-bearing. Batch-normalisation re-estimation and quantization calibration both
fit parameters to whatever data they are shown; performing either on test data fits activation
ranges to the evaluation set and renders the reported accuracy uninterpretable. This directly
addresses the validation failure identified in [7].

A secondary evaluation target is the Human Action Recognition dataset (15 classes, 12,601 labelled
images) with an ImageNet-pretrained ResNet-18 backbone, included to verify that the pipeline
generalises beyond the 32×32 CIFAR input regime to 224×224 inputs and a substantially larger
parameter count. The dataset's nominal test split carries no public labels and is therefore unusable
for measuring accuracy; a held-out portion is instead carved deterministically out of the labelled
images by sorting each class's filenames and reserving a fixed trailing fraction. Determinism by
filename rather than by seeded random number generator is chosen because the same split must be
reproduced across two independent Python and PyTorch installations, which a shared random seed does
not reliably guarantee.

### B. Network Architectures

Three architectures are used. **ResNet-20** and **ResNet-56** are the standard CIFAR residual
networks, comprising an initial 3×3 convolution followed by three stages of basic residual blocks
at widths 16, 32 and 64, with global average pooling and a single linear classifier. They are
chosen specifically *because* they are residual: the dependency-group and shortcut-alignment
problems this work addresses (Sections III-E, III-F) do not arise in a purely sequential network
and would go undetected on a VGG-style topology. **VGG-16-BN** is retained as a sequential control
for exactly that reason. **ResNet-18** serves the secondary 224×224 evaluation.

### C. System Architecture

The system is organised as a probe, four transformation stages, a measurement harness, and a search
layer that sits above all of them (Fig. 1).

![System architecture. The host probe gates the target precision; the four transformation stages each export an ONNX artifact into the measurement harness; the budget search consumes those measurements to decide whether to escalate. The test split is never used by any fitting step.](docs/architecture.png)

Two architectural invariants are enforced in code rather than documented as convention. First,
quantization is permitted only as the terminal stage; scheduling any transformation after it raises
an error at pipeline construction, formalising the ordering argument of [1]. Second, the original
uncompressed network is deep-copied at construction and preserved unmodified throughout, because it
serves simultaneously as the distillation teacher and as the accuracy reference — every in-place
stage operates on a separate copy.

### D. Mathematical Model — Notation

Let the network be a composition of layers indexed by *l* ∈ {1, …, L}. Layer *l* has weight tensor
**W**⁽ˡ⁾ ∈ ℝ^(C_out × C_in × k × k) for convolutions, or ℝ^(C_out × C_in) for linear layers. Write
*W_l* = C_out⁽ˡ⁾ for the output width of layer *l*, and **W**_c⁽ˡ⁾ for the slice of the weight
tensor associated with output channel *c*. Let *f*(·; **W**) denote the network function and
**z** = *f*(**x**) the output logits. Let *r* ∈ [0, 1) denote a requested pruning ratio and *a* ∈
{1, 2, 4, 8, 16, 32} a requested SIMD alignment.

### E. Structured Pruning: Importance and SIMD-Aligned Ratio Resolution

**Channel importance.** For output channel *c* of layer *l*, the L_p group magnitude importance is

&nbsp;&nbsp;&nbsp;&nbsp;*s*_c⁽ˡ⁾ = ‖**W**_c⁽ˡ⁾‖_p = ( Σ_{i,u,v} |*W*⁽ˡ⁾_{c,i,u,v}|^p )^{1/p} &nbsp;&nbsp;&nbsp;&nbsp;(1)

with *p* = 1 by default. Where layers are coupled by a residual connection into a dependency group
*G*, importance is aggregated across all members of the group, since they are pruned as a unit:

&nbsp;&nbsp;&nbsp;&nbsp;*s*_c^G = Σ_{l ∈ G} ‖**W**_c⁽ˡ⁾‖_p &nbsp;&nbsp;&nbsp;&nbsp;(2)

An alternative first-order loss-sensitivity criterion, following the formulation used in [2],
estimates the loss change induced by removing channel *c* via a first-order Taylor expansion about
the current weights:

&nbsp;&nbsp;&nbsp;&nbsp;Δ*L*_c⁽ˡ⁾ ≈ | Σ_{w ∈ **W**_c⁽ˡ⁾} *w* · ∂*L*/∂*w* | &nbsp;&nbsp;&nbsp;&nbsp;(3)

This criterion requires a genuine forward-backward pass over labelled data. The implementation
therefore refuses to construct a Taylor pruner without a labelled loader, rather than ranking
channels against gradients that were never populated, and re-accumulates the gradients immediately
before *every* pruning step: each step changes the weights, so a ranking carried over from a
preceding step would score a network that no longer exists. The accumulated gradients are cleared
again once the step completes, so that a subsequent recovery fine-tune does not inherit them as a
head start on its own loss.

**SIMD-aligned ratio resolution (contribution).** Realised CPU speedup depends on channel widths
remaining on vector-register boundaries, which motivates constraining the retained channel count to
a multiple of *a*. The naive implementation — and the behaviour of the underlying pruning library's
own global `round_to` parameter — applies one alignment to the whole model and floors the retained
count to the nearest lower multiple. This fails for narrow layers: when ⌊*W_l*/*a*⌋ < 2, the floor
evaluates to zero, the pruning group is rejected as invalid, and the layer silently retains *all*
of its channels while the requested ratio is still reported as applied. Relaxing the global
alignment to accommodate such a layer discards the alignment benefit for every other layer.

We instead resolve alignment per layer. With *a* a power of two, define the layer-effective
alignment *a_l* as the largest power-of-two divisor of *a* admitting at least two representable
widths, falling back to 1 when none does:

&nbsp;&nbsp;&nbsp;&nbsp;*a_l* = max ( { *a*/2^j : *j* ∈ ℕ₀, ⌊*W_l* / (*a*/2^j)⌋ ≥ 2 } ∪ {1} ) &nbsp;&nbsp;&nbsp;&nbsp;(4)

Given *a_l* > 1, the retained channel count is obtained by rounding the desired retained width to
the *nearest* achievable aligned value rather than flooring it, bounded by the layer's capacity:

&nbsp;&nbsp;&nbsp;&nbsp;*n_l* = min( ⌊*W_l*/*a_l*⌋ , max(1, round( *W_l*(1−*r*) / *a_l* )) ) &nbsp;&nbsp;&nbsp;&nbsp;(5)

&nbsp;&nbsp;&nbsp;&nbsp;*k_l* = *n_l* · *a_l* &nbsp;&nbsp;&nbsp;&nbsp;(6)

and the *effective* per-layer ratio actually applied is

&nbsp;&nbsp;&nbsp;&nbsp;*r_l*′ = 1 − *k_l* / *W_l* &nbsp;&nbsp;&nbsp;&nbsp;(7)

When *a_l* = 1 no alignment is achievable and the ratio applies directly under a clamp guaranteeing
at least one surviving channel: *k_l* = max(1, min(*W_l*, round(*W_l*(1−*r*)))).

**Clamp ordering.** The ratio entering Eq. (5) is not the raw request but min(*r*, *r*_max), where
*r*_max is the configured per-layer ceiling (0.9 by default). The ordering is load-bearing rather
than cosmetic. The underlying pruner applies that ceiling internally, *after* whatever ratio it is
handed; aligning against the raw request and letting the clamp fire afterwards would overwrite the
rounded aligned ratio with an unaligned one, silently reintroducing misalignment on exactly the
layers marked for the most aggressive pruning. Clamping first means alignment is computed against
the ratio that will actually be applied. The residual tolerance is the one already accepted for the
primary request: the realised ratio may sit up to half an alignment block away from *r*_max itself.
Once the per-layer ratios carry the alignment, the library's global `round_to` is disabled outright,
so no second, model-wide rounding rule can contradict them.

Three properties follow. Rounding to nearest rather than flooring makes |*r_l*′ − *r*| smaller in
expectation than the flooring rule. Because *a_l* is computed independently per layer, a single
narrow layer reduces alignment only for itself; every other layer retains the full *a*. And because
the rule is expressed as a per-layer ratio rather than as a model-wide rounding directive, it
composes with the per-layer overrides of Section III-G instead of competing with them. Any layer
for which *a_l* < *a*, or for which no non-trivial ratio is achievable at any alignment, is recorded
in a diagnostic warning set rather than passing silently — the failure mode this rule exists to
eliminate.

**Scope.** The resolution above presupposes that each layer's ratio is fixed and known before
pruning begins. That holds in local (per-layer) mode but not under global pruning, where the ratio a
layer receives is decided by a model-wide importance ranking at pruning time and cannot be computed
in advance. Global pruning is therefore *not* protected by this rule. It retains only a post-hoc
diagnostic, which fires when an implausibly large fraction of candidate layers emerges completely
untouched — the signature of alignment silently rejecting groups rather than of global importance
legitimately sparing them. This is detection, not repair; it is carried forward as an
acknowledged limitation in Section IV-E. The automated search of Section III-J prunes in local mode
throughout and so never enters this path.

### F. Dependency-Group Ratio Unification (contribution)

A residual addition constrains every layer feeding it to share one output channel dimension. Let
*G* denote such a dependency group, obtained from the dependency graph, and let *O* ⊆ *G* be the
subset of layers carrying an explicit ratio override (produced by sensitivity-driven protection,
Section III-G). If |{*r_l* : *l* ∈ *O*}| > 1 the group is over-determined: the underlying pruner
must assign the group a single ratio and will otherwise select one arbitrarily, silently discarding
the other. We resolve to the most protective value and propagate it across the entire group,
including members carrying no explicit override:

&nbsp;&nbsp;&nbsp;&nbsp;*r_G* = min_{l ∈ O} *r_l*,&nbsp;&nbsp;&nbsp; *r_l* ← *r_G*  ∀ *l* ∈ *G* &nbsp;&nbsp;&nbsp;&nbsp;(8)

Taking the minimum is deliberate: an override exists because sensitivity analysis identified a layer
as fragile, so resolving a conflict in the *less* protective direction would discard exactly the
information the override encodes. Unifying across the whole group — not only its overridden members
— is necessary because an un-overridden member would otherwise retain the global ratio *r* and
re-introduce the same desynchronisation at the alignment stage. Alignment (Eq. 4–7) is subsequently
computed once per group rather than once per layer, since group members share a width by
construction; rounding each member independently would re-create through alignment alone precisely
the desynchronisation Eq. (8) exists to remove.

Two implementation points fix the rule's scope. Eq. (8) is applied to every group with *O* ≠ ∅, but
a diagnostic warning is raised only in the strictly over-determined case |{*r_l* : *l* ∈ *O*}| > 1,
since that is the only case in which a value the caller explicitly requested is being overridden
rather than merely extended to its tied neighbours. And should a group's members ever be observed to
disagree on width or on resolved ratio after unification — a state the construction above should
make unreachable — the implementation falls back to independent per-layer rounding rather than
guessing which member's value ought to win.

The classifier layer is permanently excluded from pruning, as its output width equals the class
count and altering it changes what the network predicts rather than how quickly it predicts.

### G. Per-Layer Sensitivity Analysis (contribution)

Rather than searching for per-layer ratios with reinforcement learning as in [3], we measure layer
fragility directly.

**Fake quantization operator.** For a *b*-bit symmetric per-output-channel round trip, with
*q*_max = 2^(b−1) − 1, define the per-channel scale and the quantize–dequantize map:

&nbsp;&nbsp;&nbsp;&nbsp;*σ*_c = max( max_{i,u,v} |*W*_{c,i,u,v}| / *q*_max , ε ),&nbsp;&nbsp;&nbsp; ε = 10⁻¹² &nbsp;&nbsp;&nbsp;&nbsp;(9)

&nbsp;&nbsp;&nbsp;&nbsp;*Q*(**W**_c) = clamp( round(**W**_c / *σ*_c), −*q*_max−1, *q*_max ) · *σ*_c &nbsp;&nbsp;&nbsp;&nbsp;(10)

Per-channel rather than per-tensor scaling is essential: a single scale for the whole tensor is set
by its largest-magnitude filter, leaving every smaller filter using only a fraction of the available
integer range. The scaling rule mirrors the one the export path will apply (Eq. 21), so the
perturbation the analysis measures is the perturbation the deployed weights will actually carry.

It is important to state what Eq. (10) does *not* model. The map is applied to weight tensors only;
activations are left in full precision throughout, and no calibrated activation ranges (Eq. 22–23)
enter the computation. The measure is therefore a weights-only proxy for end-to-end integer
quantization, not a simulation of it, and the absolute SQNR values it produces should be read as
optimistic relative to a fully quantized graph. What the method requires, and what the proxy
supports, is the *ordering*: the relative ranking of layers by weight-induced output distortion,
which is what both downstream decisions consume. Layers whose accuracy loss originates in
activation quantization rather than weight quantization are, by construction, invisible to it.

**Distortion measurement.** Let **Z** = *f*(**X**; **W**) be the reference logits over a small
calibration batch set **X**, and **Z**⁽ˡ⁾ the logits obtained with layer *l* alone replaced by
*Q*(**W**⁽ˡ⁾), all other layers unchanged. Define the induced mean-squared distortion and the
signal-to-quantization-noise ratio in decibels:

&nbsp;&nbsp;&nbsp;&nbsp;MSE_l = (1/N) ‖**Z** − **Z**⁽ˡ⁾‖²_F,&nbsp;&nbsp; *N* = |**X**| · (number of classes) &nbsp;&nbsp;&nbsp;&nbsp;(11)

&nbsp;&nbsp;&nbsp;&nbsp;SQNR_l = 10 · log₁₀ ( ‖**Z**‖²_F / ‖**Z** − **Z**⁽ˡ⁾‖²_F ) &nbsp;&nbsp;&nbsp;&nbsp;(12)

Layers are ranked in ascending SQNR, lowest first, so that position in the ranking expresses the
output distortion a layer's compression induces. MSE_l is reported alongside as a scale-bearing
companion to the scale-free ratio of Eq. (12); only the SQNR ordering drives any decision. The
procedure requires no labels, no backpropagation and no accelerator; its cost is *L* + 1 forward
passes over a handful of calibration batches — four by default.

**Dual use of the ranking.** Let Π_k denote the *k* lowest-SQNR layers. One ranking supplies two
decisions:

1. *Quantization exclusion.* The ONNX nodes corresponding to Π_k form the exclusion set *E*, kept
   in floating point while the remainder of the graph is quantized, after expansion across residual
   additions (Eq. 26). Resolution from layer name to ONNX node name is strict: a layer that cannot
   be located in the exported graph raises an error rather than resolving to an empty exclusion
   list, which the quantizer would interpret as "exclude nothing" and which would make the
   subsequent accuracy loss inexplicable.
2. *Non-uniform pruning allocation.* Protected layers receive a scaled-down ratio,

&nbsp;&nbsp;&nbsp;&nbsp;*r_l* = *γ* · *r* if *l* ∈ Π_k \ {classifier}, otherwise *r*,&nbsp;&nbsp;&nbsp; *γ* = 0.5 &nbsp;&nbsp;&nbsp;&nbsp;(13)

yielding the non-uniform allocation that [3] obtains by policy search, at the cost of *L* + 1
forward passes rather than a full reinforcement-learning training loop. The resulting overrides are
then passed through group unification (Eq. 8) before alignment. The classifier is filtered out of
the pruning branch specifically, because it is never pruned (Section III-F) and naming it would
produce an override that goes nowhere; the exclusion branch applies no such filter, since keeping
the classifier's GEMM in floating point is a legitimate and sometimes desirable outcome.

The two consumers differ in how they are wired. The budget search applies the pruning-ratio
overrides automatically on every pruning attempt it evaluates, since the ranking is computed inside
the search loop and the allocation is what the search is optimising over. The quantization exclusion
set is not applied automatically: excluding layers from quantization trades size and latency for
accuracy in a direction the declared budget cannot infer, so it is left to the caller to supply
explicitly. The asymmetry is deliberate — the search may spend accuracy to meet a latency target,
but it may not silently decline to quantize layers the user asked to have quantized.

**Current degree of automation.** The two branches are not equally wired, and the distinction should
not be glossed. The pruning branch is consumed automatically: the budget search of Section III-J
recomputes the overrides at every pruning candidate it evaluates and passes them straight into the
pruner. The exclusion branch is implemented, tested end-to-end on a residual network, and exposed as
a caller-facing API, but the automated search issues its quantization stage with an empty exclusion
set — a user wanting sensitivity-driven exclusions must connect the two explicitly. The mechanism is
therefore available and correct but not yet automatic. Closing the gap is a matter of threading one
argument through the search loop rather than of new method, and it is listed among the open items in
Section IV-E; claims in this paper about the exclusion set should be read against that distinction.

### H. Compression-Aware Distillation

Following the teacher-reuse strategy of [1], the teacher is the original uncompressed network,
frozen for the duration; no separate teacher is trained or stored. Let **z**_s and **z**_t be
student and teacher logits, *y* the label, *T* the temperature and *σ* the softmax.

**Classical objective.** Following Hinton et al. [11]:

&nbsp;&nbsp;&nbsp;&nbsp;*L*_KD = (1 − *α*) · *L*_CE(**z**_s, *y*) + *α* · *T*² · KL( *σ*(**z**_t/*T*) ‖ *σ*(**z**_s/*T*) ) &nbsp;&nbsp;&nbsp;&nbsp;(14)

with *α* = 0.3, *T* = 4. The divergence is taken with the teacher distribution as the reference
argument, which is the direction that penalises the student for assigning low probability wherever
the teacher assigns high probability — the transfer the objective is for. The *T*² factor
compensates for the 1/*T*² attenuation that temperature softening induces in the gradient
magnitudes, keeping the two terms comparably scaled.

**Decoupled objective.** Following Zhao et al. [12], the divergence is split into a target-versus-
non-target component and a component over the ranking among non-target classes. Let *p*_τ denote
the probability assigned to the target class and define the binary collapse

&nbsp;&nbsp;&nbsp;&nbsp;**b**(**p**) = [ *p*_τ , 1 − *p*_τ ],&nbsp;&nbsp;&nbsp; *p*_τ ∈ [δ, 1−δ],&nbsp; δ = 10⁻⁷ &nbsp;&nbsp;&nbsp;&nbsp;(15)

&nbsp;&nbsp;&nbsp;&nbsp;*L*_TCKD = *T*² · KL( **b**(**p**_t) ‖ **b**(**p**_s) ) &nbsp;&nbsp;&nbsp;&nbsp;(16)

&nbsp;&nbsp;&nbsp;&nbsp;*L*_NCKD = *T*² · KL( *σ*(**z̃**_t/*T*) ‖ *σ*(**z̃**_s/*T*) ),&nbsp;&nbsp; **z̃** = **z** − M·**1**_τ &nbsp;&nbsp;&nbsp;&nbsp;(17)

&nbsp;&nbsp;&nbsp;&nbsp;*L*_DKD = *w*_CE · *L*_CE + *α* · *L*_TCKD + *β* · *L*_NCKD &nbsp;&nbsp;&nbsp;&nbsp;(18)

where **1**_τ is the target-class indicator and *M* a large constant (10³) subtracted from the
target logit *before* the softmax, so that the non-target classes renormalise to unity among
themselves rather than being merely re-weighted afterwards, with *w*_CE = 1.0, *α* = 1.0 and
*β* = 8.0. Both divergences follow Eq. (14) in taking the teacher as the reference argument. The
clamp in Eq. (15) is not cosmetic: a confident student drives *p*_τ to exactly 1 in single
precision, making the second component of **b** vanish and log(0) = −∞ propagate to NaN, after
which the pipeline would report chance accuracy as a measured result.

**Learning-rate schedule.** Fractional epoch *t* over *T*_total epochs with *T*_w warmup epochs:

&nbsp;&nbsp;&nbsp;&nbsp;*η*(*t*) = *η*₀ · *t*/*T*_w  for *t* < *T*_w; &nbsp;&nbsp; *η*₀ · ½[1 + cos(π · min((*t*−*T*_w)/(*T*_total−*T*_w), 1))] otherwise &nbsp;&nbsp;&nbsp;&nbsp;(19)

Warmup matters more here than in ordinary training: a freshly pruned network begins from a
perturbed optimum, and a full-magnitude first step risks displacing it beyond recovery.

### I. Quantization and Host Gating

**Precision gate.** Let *F* be the set of instruction-set features detected on the host and
*F*_fast = {AMX-INT8, AVX-512 VNNI, AVX-VNNI, I8MM, DotProd} the features providing a fused
low-precision dot-product path. The target precision is

&nbsp;&nbsp;&nbsp;&nbsp;*π*\* = INT8 if *F* ∩ *F*_fast ≠ ∅, else FP16 &nbsp;&nbsp;&nbsp;&nbsp;(20)

Requesting INT8 on a host failing this test raises an error unless explicitly overridden, and the
override attaches a warning to the result. This converts the hardware-dependence caveat of [5] and
[6] into an executable precondition. The gate distinguishes two failing cases when it reports:
*F* non-empty with *F* ∩ *F*_fast = ∅ means the probe read the feature set and found no fused
integer path, whereas *F* = ∅ means the probe could not read the feature set at all. Both fall back
to FP16, but only the first is a positive finding; the second is recorded as unverified rather than
as evidence that the host is slow.

The fallback is not represented as a latency win. On the ONNX Runtime CPU execution provider, FP16
weights are cast back to float32 for computation, so the FP16 path is a roughly 2× size reduction at
roughly unchanged latency. The implementation attaches exactly that statement to every FP16 result,
which is the same discipline the precision gate enforces for INT8 applied to the fallback the gate
itself selects: the system is not permitted to present its own default as faster than it is.

**Quantization maps.** Weights use symmetric per-channel quantization; activations use asymmetric
affine quantization with zero-point *z*:

&nbsp;&nbsp;&nbsp;&nbsp;*W*_q = clamp( round(*W* / *σ*_c), −128, 127 ),&nbsp;&nbsp; *σ*_c per Eq. (9) &nbsp;&nbsp;&nbsp;&nbsp;(21)

&nbsp;&nbsp;&nbsp;&nbsp;*σ*_A = (*x*_max − *x*_min) / (2^b − 1),&nbsp;&nbsp; *z* = round(−*x*_min / *σ*_A) &nbsp;&nbsp;&nbsp;&nbsp;(22)

&nbsp;&nbsp;&nbsp;&nbsp;*x*_q = clamp( round(*x* / *σ*_A) + *z*, 0, 2^b − 1 ) &nbsp;&nbsp;&nbsp;&nbsp;(23)

The activation type is chosen per architecture — signed 8-bit on ARM, unsigned on x86 — each being
the combination the respective kernels prefer. On x86 hosts lacking VNNI, the *weight* range is
additionally reduced by one bit (*q*_max → 2^(b−2) − 1 = 63) to avoid saturating the 8-bit
accumulator of the non-VNNI integer path, at the cost of one bit of weight resolution. The
correction is applicable only to that platform class: VNNI-capable cores accumulate full 8-bit
products without overflow, and on ARM the reduction would discard precision for nothing. The
reduction applies to weights rather than to activations, and the reported quantity therefore
degrades weight fidelity, not activation range.

**Operator coverage audit (contribution).** In QDQ representation an operator is fused into an
integer kernel only when bracketed on *both* sides. Let *V* be the set of compute nodes
(convolution, transposed convolution, GEMM, matrix multiply). Define

&nbsp;&nbsp;&nbsp;&nbsp;*v* ∈ *V*_q ⟺ (∃ input of *v* produced by DequantizeLinear) ∧ (∃ output of *v* consumed by QuantizeLinear) &nbsp;&nbsp;&nbsp;&nbsp;(24)

&nbsp;&nbsp;&nbsp;&nbsp;Coverage = |*V*_q| / |*V*| &nbsp;&nbsp;&nbsp;&nbsp;(25)

Checking only the input side would count an *excluded* node as quantized, because it still consumes
the dequantize node closing the preceding pair — inflating the very metric intended to expose
"smaller but not faster." Eq. (24) covers the QDQ representation the pipeline emits; nodes already
rewritten to an explicitly integer operator type (`QLinearConv`, `MatMulInteger` and their kin)
count as quantized without the bracketing test, since for those the integer kernel is the operator
rather than a fusion of it. Any operator left in floating point is reported explicitly, by type and
count, alongside the size reduction it failed to convert into speed.

**Residual exclusion propagation.** The runtime decides QDQ placement from consumer requirements
rather than from the exclusion list alone, so naming only a convolution that feeds a residual
addition is silently ineffective: the addition still requests quantized inputs, and the quantizer
supplies them by quantizing the convolution anyway. For each excluded node *n*, the exclusion set is
therefore expanded across one hop to the additions *n* feeds:

&nbsp;&nbsp;&nbsp;&nbsp;*E*′ = *E* ∪ { *m* : ∃ *n* ∈ *E*, *m* ∈ consumers(*n*), optype(*m*) = Add } &nbsp;&nbsp;&nbsp;&nbsp;(26)

Two properties of Eq. (26) are confirmed by regression test on ResNet-20 rather than argued. The
expansion is sufficient: a convolution named in *E* that feeds a residual addition is reported as
still floating point after quantization, where before the expansion it was quantized in spite of
being named. And the expansion is not over-broad: a residual addition in a downsampling block takes
two convolutional inputs — the main path and the shortcut — and excluding only the main path's
convolution leaves exactly one operator in floating point, the shortcut branch being quantized
normally rather than swept up as collateral. Eq. (26) is deliberately one hop deep and scoped to
`Add`; Section IV-E records what that scoping does and does not cover.

### J. Budget-Driven Search

Let a budget *B* = (*τ*_lat, *δ*_acc, *α*_min, *τ*_size) specify upper bounds on latency, accuracy
drop and size, and a lower bound on accuracy. A candidate measurement *m* against baseline *m*₀
satisfies *B* when all active constraints hold. Search proceeds by escalation:

```
Algorithm 1: Budget-driven compression search
─────────────────────────────────────────────
Input : model f, calibration Xc, test Xt, budget B, alignment a
Output: least-aggressive configuration satisfying B

 1  H ← probe_host();  π* ← precision_gate(H)              ▷ Eq. (20)
 2  m₀ ← measure(export(f))                                 ▷ baseline
 3  m  ← measure(quantize(f, π*))                           ▷ attempt 1
 4  if satisfies(B, m₀, m) then return remeasure_full(m)
 5
 6  R ← sensitivity_analyze(f, Xc)                          ▷ Eq. (9)–(12)
 7  if latency_or_size_violated(B, m₀, m) then              ▷ attempt 2
 8      lo ← 0.0 ;  hi ← 0.9
 9      for step ← 1 to 4 do
10          r ← (lo + hi)/2
11          ρ ← unify_groups(protect(R, r))                 ▷ Eq. (8), (13)
12          ρ′← align_per_layer(ρ, a)                       ▷ Eq. (4)–(7)
13          m ← measure(quantize(recover(prune(f, ρ′)), π*))
14          if latency_satisfied(B, m) then
15              best ← m ;  hi ← r
16              if satisfies(B, m₀, m) then return remeasure_full(m)
17          else lo ← r
18
19  t ← best ?? hardest_tried ?? attempt₁          ▷ recovery target
20  if accuracy_violated(B, t) and train data available then     ▷ attempt 3
21      m ← measure(quantize(distill(recover(prune(f, ρ′_t)), f), π*))
22      if satisfies(B, m₀, m) then return remeasure_full(m)
23
24  F ← { m ∈ attempts : satisfies(B, m₀, m) }
25  return remeasure_full( argmax_{m ∈ (F if F ≠ ∅ else attempts)} E(m₀, m) )  ▷ Eq. (28)
```

Escalation order is not arbitrary. Quantization is attempted first because it requires no
retraining and no architectural change. Pruning is added only when latency or size is the violated
constraint — if quantization already met the latency target and accuracy is the shortfall, pruning
would worsen precisely the failing quantity. Distillation is added last, as its function is
recovering accuracy the earlier stages cost.

Two details of the fallback path in lines 19 and 24–25 are worth stating, because they determine
what the search returns when it fails. Distillation recovers the *gentlest* ratio that met the
latency target if one exists; if none did, it recovers the most aggressive ratio actually tried
rather than silently abandoning pruning, since that candidate is the closest the search came to the
latency constraint. And the final ranking is taken over budget-satisfying attempts when any exist,
falling back to the full attempt set only when none does — so a configuration that meets the budget
is never passed over for a higher-scoring one that does not.

One limitation of the loop as implemented should be read off Algorithm 1 directly rather than left
to inference. The sensitivity report *R* computed at line 6 is consumed only through `protect` at
line 11, which yields pruning-ratio overrides; the quantization calls at lines 3, 13 and 21 are
issued with an empty exclusion set. The dual use described in Section III-G is therefore only half
realised inside the search.

**Ranking metric.** Adapting the efficiency score of [1], with the substitution of *measured
latency* for the original's floating-point operation count:

&nbsp;&nbsp;&nbsp;&nbsp;*C*(*m*₀, *m*) = ½ [ max(0, 1 − size(*m*)/size(*m*₀)) + max(0, 1 − lat(*m*)/lat(*m*₀)) ] &nbsp;&nbsp;&nbsp;&nbsp;(27)

&nbsp;&nbsp;&nbsp;&nbsp;*E*(*m*₀, *m*) = 2 · *A*(*m*) · *C*(*m*₀, *m*) / ( *A*(*m*) + *C*(*m*₀, *m*) ) &nbsp;&nbsp;&nbsp;&nbsp;(28)

where *A*(*m*) is top-1 accuracy. The harmonic mean is the operative choice: a configuration that
preserves accuracy while compressing nothing, and one that compresses aggressively while destroying
accuracy, both score near zero. The substitution in Eq. (27) is the point of departure from [1] —
the original metric is computable without executing the model, and therefore cannot detect the
smaller-but-slower outcome that motivates this work.

### K. Measurement Protocol

Every stage exports an ONNX artifact and is measured identically. Latency is reported as the median
and 95th percentile over 200 timed single-image forward passes following 50 discarded warmup
iterations, with the session pinned to one intra-operator and one inter-operator thread. All
precisions are timed through an identical session configuration; comparing precisions measured
under differing thread counts or batch sizes is the most direct route to a fabricated speedup.
Accuracy is computed on the exported artifact rather than the in-memory module, so that export
defects and quantization error appear in the reported figure instead of being concealed. Size is
the actual byte count on disk. During search, accuracy is estimated on a subset and latency over
fewer iterations for tractability; the selected configuration is then re-measured at full fidelity,
so that reported figures are never the search-time estimates.

**Classification metrics.** For class *c* with true positives TP_c, false positives FP_c and false
negatives FN_c:

&nbsp;&nbsp;&nbsp;&nbsp;*P*_c = TP_c/(TP_c+FP_c), &nbsp; *R*_c = TP_c/(TP_c+FN_c), &nbsp; *F*1_c = 2*P*_c*R*_c/(*P*_c+*R*_c) &nbsp;&nbsp;&nbsp;&nbsp;(29)

reported as unweighted macro averages over the ten classes, appropriate here since CIFAR-10 is
class-balanced. The full confusion matrix from which Eq. (29) is derived is accumulated in the same
evaluation pass that produces top-1, so the per-class breakdown costs no additional inference, and
macro averages are taken over the classes actually observed rather than over all ten — a
distinction that matters under the truncated evaluation used during search, and one whose
consequences for interpretation are discussed in Section IV-C.

---

## IV. Expected Results and Discussion

This section states the outcomes the protocol of Section III-K is designed to test. With the
exception of the floating-point baseline in Section IV-A, which is measured, all figures below are
*projections* derived from the surveyed literature and from the structural properties of the
method; they are stated as falsifiable expectations, not as findings.

### A. Measured Baseline

The ResNet-56 floating-point baseline was trained on CIFAR-10 for 60 epochs under a cosine schedule
with warmup, reaching **93.18% top-1 accuracy** on the held-out 10,000-image test split (best epoch
coincident with final epoch; 45.8 minutes wall-clock). This is the reference against which all
compression configurations are compared, and it is consistent with commonly reported ResNet-56
CIFAR-10 baselines, supporting the validity of the training procedure.

### B. Expected Compression Outcomes

The sweep crosses pruning ratios *r* ∈ {0.0, 0.3, 0.5, 0.7} with precisions {FP32, INT8}, with and
without distillation recovery. Expected behaviour:

**TABLE II. PROJECTED SWEEP OUTCOMES (ResNet-56 / CIFAR-10) — TO BE POPULATED BY MEASUREMENT**

| Configuration | Top-1 (%) | Δ vs. baseline (pp) | Median latency | Size | Efficiency *E* |
|---------------|-----------|---------------------|----------------|------|----------------|
| Baseline FP32 | 93.18 (measured) | — | *to measure* | *to measure* | — |
| INT8 only | expect −0.3 to −1.0 | small | precision-gate dependent | ≈ ¼ baseline | *to compute* |
| Prune 0.3 + recover + INT8 | expect −1 to −2 | moderate | expect ↓ | ↓↓ | *to compute* |
| Prune 0.5 + recover + INT8 | expect −2 to −4 | moderate | expect ↓↓ | ↓↓ | *to compute* |
| Prune 0.5 + DKD + INT8 | expect −0.5 to −2 | recovers most of the above | as above | as above | *to compute* |
| Prune 0.7 + DKD + INT8 | expect −4 or worse | large | expect ↓↓↓ | ↓↓↓ | *to compute* |

The expected accuracy trajectory — monotonic decline with pruning ratio, partially recovered by
distillation — follows the pattern reported in [1], where top-1 declined from 71.84% to 63.70% at
*r* = 0.5 on the harder CIFAR-100 task, and in [9], where distillation recovered a pruned CIFAR-10
student to 75.96% from 71.38% trained from scratch. Because CIFAR-10 is a substantially easier task
than CIFAR-100, we expect proportionally smaller degradation at matched ratios.

### C. Expected Precision, Recall and Per-Class Behaviour

Macro-averaged precision and recall (Eq. 29) are expected to track top-1 accuracy closely at low
pruning ratios, since CIFAR-10 is class-balanced and the network is not operating in a
long-tailed regime. The informative expectation is *divergence at high ratios*: as capacity is
removed, per-class recall should degrade non-uniformly, with visually confusable class pairs — cat
and dog, automobile and truck — absorbing a disproportionate share of the loss. This expectation is
the direct analogue, at the class level, of the per-task collapse observed in [4], where an
aggregate improvement concealed one benchmark falling from 73.8% to 24.5%. Reporting macro
precision and recall alongside accuracy is precisely what makes such a collapse visible; the
per-class breakdown is therefore a reporting requirement rather than an optional addition.

The measurement harness accumulates the full confusion matrix for each stage in the same evaluation
pass that produces top-1 accuracy, so per-class precision, recall and F1 are available at no
additional inference cost. One detail of the averaging is worth stating explicitly, because it
affects how the reported figures should be read: macro averages are taken over those classes that
actually appear in the evaluated split. During the budget search the test set is deliberately
subsampled for tractability, and a truncated pass over an unshuffled split can omit whole classes;
averaging a zero in for a class that was never evaluated would understate macro recall by an amount
determined by the truncation rather than by the model. Configurations evaluated on a partial split
are flagged as such, and the selected configuration is re-measured over the full test set before
its figures are reported.

### D. Expected Effect of the Contributed Mechanisms

**Precision gating.** On a host lacking fused integer kernels, INT8 is expected to be approximately
4× smaller and *no faster, possibly slower*, than FP32. This is the outcome the gate exists to
predict in advance, and confirming it on such a host constitutes positive evidence for the gate
rather than a negative result — it reproduces, under controlled measurement, the effect [5] reports
qualitatively.

**SIMD alignment.** Ablating the per-layer rule (Eq. 4–7) against a single global alignment is
expected to show a subset of narrow layers retaining 100% of their channels under the global rule
while the requested ratio is nominally applied. The diagnostic output should name those layers; the
measured parameter reduction under the global rule should fall short of the requested ratio by a
margin attributable exactly to them.

**Dependency-group unification.** On the residual architectures evaluated, the overrides of Eq. (13)
will frequently land on only part of a tied group, since the sensitivity ranking is computed per
layer and knows nothing of the dependency graph. The expectation is therefore observational rather
than comparative: unification should be seen to fire on ResNet-56 at every ratio, extending each
override to its tied neighbours, and the realised widths of every member of a tied group should be
identical. A run in which two members of one group emerge with different widths would falsify the
mechanism directly. The mechanism is inert on VGG-16-BN, which is why that architecture is retained
as a sequential control.

**Sensitivity-guided allocation.** Protecting the two lowest-SQNR layers (Eq. 13) is expected to
recover accuracy relative to a uniform ratio at matched overall compression. The expected magnitude
is modest — on the order of a fraction of a point to a few points — consistent with the 0.3-point
margin that [3] actually achieves over hand-tuned baselines once its table is read rather than its
abstract. This expectation covers the pruning branch only; the corresponding test for the
quantization-exclusion branch requires the manual wiring described in Section III-G and is not part
of the automated sweep.

**Operator coverage.** Coverage (Eq. 25) below unity would indicate compute operators left in
floating point, and the accompanying latency measurement should show the corresponding shortfall
against the size reduction. A configuration exhibiting a large size reduction alongside low coverage
and unchanged latency is the failure mode this audit is designed to surface. The audit also provides
the observable for the exclusion mechanism of Eq. (26): a deliberately excluded convolution should
appear in the still-floating-point count, and its absence from that count — coverage remaining at
unity despite a non-empty exclusion set — is precisely the silent failure the expansion corrects.

### E. Threats to Validity

Six limitations are acknowledged in advance, three concerning the scope of the contributed
mechanisms and three concerning the interpretation of the measurements.

*Scope of the mechanisms.* The pruning module's alignment and grouping logic has not been exercised
against grouped or depthwise-separable convolutions, since none of the evaluated architectures
employ them; because such layers do not admit independently prunable channels, the per-layer
alignment resolution of Eq. (4)–(7) cannot be assumed to transfer, and MobileNet-family networks
would require separate validation. The per-layer alignment protection applies to local pruning only:
under global pruning the ratio each layer receives is not known before pruning begins, so Eq. (4)–(7)
cannot be evaluated in advance and the implementation falls back to a post-hoc diagnostic that
reports the symptom without preventing it. This is a genuine gap rather than a design choice,
mitigated at present only by the fact that the automated search never selects global mode. And the
residual exclusion propagation of Eq. (26) is single-hop and scoped to addition operators. This is
sufficient for the evaluated topologies — no residual sum in these architectures feeds directly into
another, a ReLU always intervening — but it is not a general transitive closure, and networks that
merge branches through concatenation or multiplicative attention are not covered at all.

*Interpretation of the measurements.* The sensitivity measure of Eq. (9)–(12) perturbs weights only.
It is a proxy for the distortion a fully quantized graph would exhibit, not a simulation of one, and
a layer whose fragility originates in activation quantization is invisible to the ranking; the
protective decisions the ranking drives inherit that blind spot. The sensitivity ranking's
quantization-exclusion branch is not consumed by the automated search (Section III-J), so any result
obtained from the sweep tests the pruning-allocation half of the dual-use claim and not the
exclusion half — the latter requires a separately constructed pipeline. And all latency figures are
host-specific by construction: that is the design intent, but it means absolute timings are not
comparable across machines, only the ratios measured on one host under an identical session
configuration.

---

## V. Conclusion

This paper has presented the design and mathematical formulation of a compression toolkit for
convolutional neural networks whose organising principle is that compression claims must be
verified on the deployment runtime rather than inferred from model structure. The motivation was
established empirically from the literature: across ten surveyed works spanning pipeline
integration, structured pruning, quantization and knowledge distillation, only two report measured
on-device latency, while the remainder motivate themselves with deployment efficiency and report
exclusively floating-point operation counts, parameter counts or file sizes. The same survey
identified two further recurring weaknesses — headline figures drawn from best-case configurations
rather than averages, and, in the most aggressive compression regime, validation against training
rather than held-out accuracy, with a documented gap of nearly 38 points between the two.

Four mechanisms were contributed in response. The per-layer SIMD-aligned ratio resolution rule of
Equations (4)–(7) constrains retained channel counts to vector-register-aligned multiples while
computing the achievable alignment independently for each layer, eliminating a failure mode in which
a single narrow layer either silently retains all of its channels or forces the relaxation of
alignment across the entire network. Dependency-group ratio unification, Equation (8), resolves the
over-determination that arises when a residual connection ties layers assigned differing pruning
ratios into a group constrained to share one channel dimension, resolving toward the most
protective ratio because an override encodes a measured fragility that the alternative would
discard. The signal-to-quantization-noise sensitivity measure of Equations (9)–(13) ranks layers by
the weight-quantization distortion their compression induces at the output, requiring neither
labels, gradients nor an accelerator, and supplies a single ranking to two distinct decisions — the
quantization exclusion set and the non-uniform pruning allocation — thereby obtaining at the cost of
*L* + 1 forward passes the non-uniform allocation that prior work derives from
reinforcement-learning policy search. Of those two consumers, the pruning allocation is wired into
the automated search as described; the exclusion set is implemented and verified but must at present
be connected by the caller, a gap recorded in Section IV-E and revisited below.
Finally, the host-gated precision policy of Equation (20) and the operator-coverage audit of
Equations (24)–(25) together make it structurally impossible for the system to report a file-size
reduction as a latency improvement: the former refuses integer quantization on hosts without a
fused low-precision kernel path, and the latter counts only those compute operators genuinely
bracketed by quantize–dequantize pairs on both sides, a distinction that a naive input-side check
would obscure precisely where the audit matters most.

These mechanisms are composed under a budget-driven search that escalates from quantization alone
to structured pruning and then to distillation recovery only while a declared constraint remains
unsatisfied, terminating at the least aggressive configuration that suffices. The escalation order
follows from the objective rather than from convention: pruning is introduced only when latency or
size is the violated quantity, since introducing it when accuracy is the shortfall would worsen the
failing constraint, and distillation is introduced last because its function is recovering accuracy
the preceding stages cost. Configurations are ranked by an efficiency score adapted from the
surveyed literature, with measured latency substituted for the original formulation's operation
count — a substitution that is the crux of the present approach, since the original metric is
computable without ever executing the model and therefore cannot, in principle, detect the
smaller-but-slower outcome this work exists to prevent.

The floating-point baseline has been established at 93.18% top-1 accuracy for ResNet-56 on
CIFAR-10. The compression sweep specified in Section IV-B, under the measurement protocol of
Section III-K, remains to be executed, and the
quantitative claims of Section IV are stated as falsifiable expectations rather than results. Three
of those expectations are of particular interest because they are capable of disconfirming the
design: that integer quantization on a host failing the precision gate will prove smaller but not
faster; that ablating per-layer alignment in favour of a single global value will leave an
identifiable subset of narrow layers wholly unpruned while the requested ratio is nominally
reported as applied; and that macro-averaged precision and recall will diverge from top-1 accuracy
at high pruning ratios in a class-dependent pattern, revealing degradation that an aggregate
accuracy figure alone would conceal.

Future work follows directly from the acknowledged limitations. Validation against grouped and
depthwise-separable convolutions is required before the alignment logic can be considered general,
as the MobileNet family — the most common target for edge deployment — is precisely the case the
current implementation has not exercised. The residual exclusion propagation should be generalised
from its present single-hop, addition-scoped form to a transitive closure over arbitrary combining
operators, including concatenation and multiplicative attention-style merges. Extending the
per-layer alignment protection to global pruning requires a different formulation than Eq. (4)–(7),
since no fixed per-layer ratio exists to align against; the natural approach is to intercept the
pruner's per-group decisions as they are made rather than to precompute them, which would replace
the present post-hoc diagnostic with an actual guarantee. The quantization-exclusion branch of the
sensitivity ranking should be threaded through the budget search so that the dual-use claim holds
automatically rather than only under manual assembly — a small change, but one the paper's claims
depend on and which should not be represented as complete until made. Finally, the toolkit
currently performs post-training quantization only, and its half-precision fallback is a size
reduction rather than a latency one on the CPU execution provider;
incorporating quantization-aware training, with the original network reused as teacher in the
manner the surveyed pipeline literature validates, would allow the system to trade additional
training cost for accuracy retention at low bit-widths, and would complete the correspondence
between the pipeline implemented here and the composition that the literature identifies as
best-performing.

---

## References

[1] W. Chae and K. Seo, "Pipeline of Pruning, Knowledge Distillation, and Quantization for Model
Compression," *Journal of Electrical Engineering & Technology*, vol. 21, pp. 2099–2105, 2026, doi:
10.1007/s42835-025-02541-7.

[2] Z. Wang, L. Shi, Z. Mei, X. Zhao, Z. Wang, and J. Li, "Iterative knowledge distillation and
pruning for model compression in unsupervised domain adaptation," *Pattern Recognition*, vol. 164,
art. 111512, 2025, doi: 10.1016/j.patcog.2025.111512.

[3] Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han, "AMC: AutoML for Model Compression and
Acceleration on Mobile Devices," in *Proc. European Conference on Computer Vision (ECCV)*, 2018.

[4] M. Zhou, R. Mao, S. Sripada, and X. Li, "AEMC: Aggregator-efficient model compression for
streamlining LLMs," *Neurocomputing*, vol. 672, art. 132779, 2026, doi:
10.1016/j.neucom.2026.132779.

[5] W. Chen, H. Qiu, J. Zhuang, C. Zhang, Y. Hu, Q. Lu, T. Wang, Y. Shi, M. Huang, and X. Xu,
"Quantization of Deep Neural Networks for Accurate Edge Computing," *ACM Journal on Emerging
Technologies in Computing Systems*, vol. 17, no. 4, art. 54, 2021, doi: 10.1145/3451211.

[6] R. Abdel-Salam, A. H. Abdel-Gawad, and A. G. Wassal, "VLCQ: Post-training quantization for deep
neural networks using variable length coding," *Future Generation Computer Systems*, vol. 166, art.
107654, 2025, doi: 10.1016/j.future.2024.107654.

[7] N. A. Khan and A. M. S. Rafat, "Optimizing Deep Learning Models for Resource-Constrained
Environments With Cluster-Quantized Knowledge Distillation," *Engineering Reports*, vol. 7, art.
e70187, 2025, doi: 10.1002/eng2.70187.

[8] X. Fang, L. Chen, H. Yin, X. Chen, and W. Wang, "Communication Efficient Federated Learning
With Quantization-Aware Training Design," *IEEE Transactions on Machine Learning in Communications
and Networking*, vol. 4, pp. 45–59, 2026, doi: 10.1109/TMLCN.2025.3635050.

[9] Y. Chen, C. Li, L. Gong, X. Wen, Y. Zhang, and W. Shi, "A deep neural network compression
algorithm based on knowledge transfer for edge devices," *Computer Communications*, vol. 163, pp.
186–194, 2020, doi: 10.1016/j.comcom.2020.09.016.

[10] L. Xu, J. Ren, Z. Huang, W. Zheng, and Y. Chen, "Improving Knowledge Distillation via Head and
Tail Categories," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 34, no. 5,
pp. 3465–3478, 2024, doi: 10.1109/TCSVT.2023.3325814.

[11] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," in *NIPS
Deep Learning and Representation Learning Workshop*, 2015, arXiv:1503.02531.

[12] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled Knowledge Distillation," in *Proc.
IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11953–11962.
