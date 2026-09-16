# Literature Survey: Neural Network Compression via Pruning, Quantization, and Knowledge Distillation

## 1. Introduction

Pruning, quantization, and knowledge distillation (KD) are the three dominant approaches to
reducing the memory footprint and inference cost of convolutional neural networks (CNNs) for
deployment on resource-constrained hardware. Historically these techniques were studied in
isolation; more recent work has begun to examine how they interact when combined into a single
pipeline. This survey reviews ten papers spanning pipeline integration [1], [2], structured
pruning [3], [4], quantization [5]–[8], and knowledge distillation [9], [10], with particular
attention to the specific quantitative claims each paper makes and to methodological gaps that
limit how far those claims can be generalized. This critical reading directly informs the design
of a prune → distill → quantize compression toolkit, since several recurring gaps — reliance on
FLOPs/size proxies rather than measured latency, best-case rather than average-case reporting,
and train/test conflation at extreme compression ratios — are exactly the failure modes such a
toolkit needs to validate against, not merely inherit.

## 2. Compression Pipeline Ordering and Integration

Chae and Seo [1] address the question of *how* to combine pruning, KD, and quantization rather
than studying any one technique alone. They argue that quantization must be applied last, since
converting weights to a lower-precision datatype (e.g., INT8) is difficult to reverse and
complicates further gradient-based pruning or distillation. Their proposed pipeline is
sequential: prune → knowledge-distillation-assisted fine-tuning → quantization-aware training
(QAT) → INT8 export. Central to their contribution is Compression-Aware Knowledge Distillation
(CAKD), which reuses the original, uncompressed model as the teacher at two points in the
pipeline — during post-pruning fine-tuning and again during QAT — removing the need to train or
store a separate teacher network. They also propose an Efficiency Score, defined as the harmonic
mean of accuracy and a compression score (itself the mean of FLOPs-drop and size-drop ratios), as
a single metric for comparing pipeline configurations. Evaluated on ResNet-56/CIFAR-100 and
MobileNetV2/Tiny-ImageNet, the pipeline's reported results are consistently FLOPs- or size-based;
GPU hardware mentioned in the paper is used only for training, not for measuring inference
latency, and the two FLOPs tables corresponding to the same configuration are internally
inconsistent — a caution against treating the paper's proxy metrics as a stand-in for measured
on-device latency.

Wang et al. [2] tackle a related but distinct question: given that pruning and KD will be
combined, should they be applied once each, or interleaved? Their method (ITMC) alternates
Transfer Knowledge Distillation (TKD) and Adaptive Channel Pruning (ACP) every epoch rather than
pruning once and distilling once. Channels are ranked by a first-order Taylor loss-sensitivity
criterion and removed in small batches of 16, with TKD fine-tuning applied immediately after each
removal step, within a scheduled window of epochs whose pruning rate ramps up and back down.
Across Office-31, ImageCLEF-DA, and PACS domain-adaptation benchmarks with VGG19→VGG16 and
ResNet101→ResNet50 teacher-student pairs, ITMC is reported to outperform baselines by margins as
large as +11.2 and +18.6 percentage points on individual transfer tasks. However, these headline
figures are best-single-task comparisons against the *weakest* baseline, not averages: the
paper's own aggregate table shows a more modest average gain of +1.3 points over the uncompressed
base model and +4.4 points over the strongest baseline. As with [1], no latency or throughput is
measured despite the paper motivating itself with "real-time, resource-constrained" deployment
scenarios — only FLOPs and accuracy are reported.

## 3. Structured Pruning

He et al.'s AMC [3] automates the choice of per-layer pruning ratio using a DDPG reinforcement
learning agent, replacing hand-tuned uniform sparsity with a learned, non-uniform policy. Each
layer is represented as an 11-dimensional state (including layer type, input/output channel
counts, spatial size, stride, kernel size, remaining FLOPs budget, and the previous layer's
action), and the agent outputs a continuous sparsity ratio per layer, validated without
fine-tuning as a fast reward proxy before a final fine-tuning pass. On ImageNet, AMC compresses
ResNet-50 by 3.4–5× with top-1/top-5 accuracy of 76.11%/92.89% versus a 76.13%/92.86% baseline,
and prunes MobileNet-V1 to 285 MFLOPs at 70.5% top-1 — better than a 0.75×-width baseline at 325
MFLOPs (68.4%) — while yielding a 1.53× (GPU) to 1.95× (Pixel 1) measured speedup, one of the few
papers in this survey to report on-device latency directly. Its central finding — that
non-uniform, learned per-layer pruning ratios outperform hand-tuned uniform ratios — is
well-supported. However, the abstract's headline claim of a "2.7% improvement over hand-crafted
methods under 4× FLOPs reduction" does not reconcile with the paper's own results table, which
reports VGG-16 results only at 5× (20% FLOPs) reduction, where the margin over the best
hand-crafted baseline is 0.3 points, not 2.7.

Zhou et al.'s AEMC [4] identifies a structural failure mode directly relevant to residual-network
pruning: removing contiguous blocks from a residual architecture disrupts the trained ratio
between the shortcut and residual paths ("amplitude mismatch"), because residual-stream direction
stays stable with depth while its norm grows monotonically, and later blocks are tuned to expect
a particular norm. Their fix is a parameter-free, per-layer scalar rescaling — calibrated from as
few as three to four examples on the dense model, with no gradient updates — applied in place of
each pruned block, genuinely training-free per their own reported memory-overhead measurements.
On LLaMA-3.1-8B at 24.4% pruning, this recovers accuracy from a 34.0% (SliceGPT) or 39.1%
(training-free LLM-Streamline) baseline up to 56.3%, against a 61.7% dense-model ceiling. This
result is architecturally relevant to CNN pruning even though the paper's target is LLMs, since
the amplitude-mismatch diagnosis is a general property of residual connections, not specific to
transformer blocks. Its main limitation is that the aggregate accuracy gain masks per-task
collapse — on one benchmark (ChID), AEMC's accuracy falls from 73.8% (dense) to 24.5%, barely
above the 19–21% baselines it is framed as fixing — indicating the amplitude correction mitigates
but does not eliminate degradation on every task, and all reported efficiency gains are
memory-based, with no latency measurement despite motivating the method via decoding speed.

## 4. Quantization

Chen et al. [5] study how compression accuracy varies with bit-width across segmentation,
classification, and speech-recognition tasks, using incremental (power-of-two) quantization from
2 to 9 bits. Their central claim — that quantization can act as a regularizer on
over-parameterized models, with a mid-range bit-width (6–8 bit) often outperforming both very low
and very high bit-widths — holds clearly for their segmentation task (peak accuracy of
86.51–86.66% at 6–7 bits versus 85.26–86.12% at 9 bits) but is contradicted by their own speech
recognition results, where 9 bits is the best-performing configuration outright. The paper
reports memory reductions of 3.5–6.4× but explicitly notes that no speedup is observed on
unmodified CPU/GPU hardware, citing a separate prior work for FPGA-only speedup figures — a
direct caution that INT8 memory savings do not imply INT8 latency savings without matching
hardware kernels. Two of the paper's own result tables mislabel error rates as accuracy
improvements (reporting a higher top-1 *error* value as an "improvement"), an internal labeling
inconsistency worth noting when citing its numbers directly.

Abdel-Salam et al.'s VLCQ [6] is a data-free post-training quantization method: weight tensors are
split into negative and positive regions, each quantized with an independent symmetric scale, and
the resulting integers are further compressed with a modified Golomb-Rice variable-length code
that exploits the near-zero-heavy shape of typical weight distributions. Under one specific
configuration (per-channel quantization, ResNet18, 16 bins), this reaches a 12.26× effective
compression ratio at a 3.89-point top-1 drop (65.87% vs. 69.76% FP32 baseline) — matching the
paper's headline "12.25× within 4% loss" claim, though this figure describes a single best-case
configuration rather than a typical result; the paper's more general per-channel claim across
models is a softer "up to 6× vs. FP32." The authors themselves acknowledge, in their discussion
section, that the scale-search heuristics are static rather than adaptive to bit-width or
architecture, and that their Golomb-Rice decoder's throughput claim ("should not hurt throughput")
is asserted, not measured — no latency or cycle-count result is reported for the decoding step.

Khan and Rafat [7] propose Cluster-Quantized Knowledge Distillation (CQKD), which combines
structured L1-norm filter pruning with layer-wise K-means quantization (weights replaced by
learned centroids) applied to both teacher and student, evaluated on a modified ResNet-18 for
CIFAR-10/100. The paper reports extreme compression ratios (34,052× on CIFAR-10, 34,193× on
CIFAR-100). A significant gap emerges on closer reading: the composite KD loss described as a
core contribution is presented only as a hypothetical formulation and never implemented in the
reported experiments, which instead use plain cross-entropy training for the student — meaning
the results do not actually demonstrate the paper's namesake distillation mechanism. In addition,
the abstract's headline accuracy figures (97.9%/91.2%) are training accuracies; the paper's own
results table reports substantially lower test accuracies of 87.39% (CIFAR-10) and 53.61%
(CIFAR-100), a roughly 10-and 38-point train/test gap that is not disclosed in the abstract — a
concrete illustration of why extreme compression claims must always be checked against held-out
test performance rather than training accuracy.

Fang et al. [8] propose QAT-FL for federated learning, in which each client performs ordinary
local training followed by additional epochs with a fake-quantization module (straight-through
estimator) inserted before communicating quantized weight updates to the server. On MNIST and
CIFAR-10 under i.i.d. simulation, QAT-FL outperforms post-training quantization (PTQ) by roughly
2–9 and 2–4 percentage points respectively, and the paper reports a proof of convergence. A
notable limitation is that the convergence bounds derived for QAT-FL and standard PTQ-FL are
algebraically identical in the paper's own equations; QAT-FL's claimed advantage rests on an
asymmetric relaxation applied to one term in an appendix derivation rather than a demonstrated
numerical gap between the two bounds. Additionally, the paper's one non-i.i.d., realistic setting
(FEMNIST) is evaluated only with qualitative accuracy-curve claims, not the quantitative tables
used elsewhere, leaving the method's advantage under real data heterogeneity unverified against
concrete numbers.

## 5. Knowledge Distillation

Chen et al. [9] combine architectural lightweighting (replacing fully-connected layers with
global average pooling and standard convolutions with separable convolutions) with a two-part
distillation loss — a middle-layer feature-distance term and a logits-layer KD term in which the
teacher's soft targets are perturbed with Gaussian noise before the softmax, acting as a
regularizer — followed by global magnitude-based iterative pruning (2% per iteration across 15
iterations, with fine-tuning between steps). On CIFAR-10 with a VGG-based teacher/student pair,
their method reaches 75.96% accuracy versus 74.54% for standard KD and 71.38% for a
student-trained-from-scratch. Checking the paper's own reported parameter counts against its
headline claims reveals two discrepancies: the actual compression ratio implied by the reported
parameter counts (39.24M → 0.81M) is 48.4×, not the abstract's stated ">49.5×," and the claimed
">3.2× speedup on Raspberry Pi 3" cannot be attributed to that platform specifically, since the
paper states the uncompressed teacher could not run on the Raspberry Pi 3 at all due to memory
overflow — the only measurable teacher-vs-compressed speedups (4.2–4.8×) come from GPU and a
Cambricon accelerator, not the edge device the claim is associated with.

Xu et al.'s ReKD [10] decouples the standard KD loss by rank: for each sample, the teacher's
top-K logits ("head" categories) and the remaining logits ("tail" categories) are each converted
into a separate KL-divergence term (TISD and NTID respectively) with independently tunable
weights, motivated by an upper-bound derivation for which the full proof is deferred to
supplementary material. Across CIFAR-100, Tiny-ImageNet, ImageNet-1K, SVHN, and MS-COCO detection,
ReKD consistently improves over classical KD, with gains ranging from roughly +0.6 to +3.7
percentage points depending on the dataset and architecture pair. The paper explicitly states
that "ReKD does not introduce additional hyperparameters and computational overhead," but this is
contradicted by its own method and ablation: the approach introduces three new tunable
hyperparameters (K, and the two loss weights) absent from classical KD, and its own ablation shows
the optimal K is dataset-dependent (K=7 for Tiny-ImageNet vs. K=3 for CIFAR-100), requiring a
per-dataset search rather than being overhead-free; furthermore, despite the no-overhead claim
appearing three times in the text, no FLOPs, runtime, or latency measurement is reported anywhere
in the paper to substantiate it.

## 6. Cross-Cutting Methodological Gaps

Several patterns recur across these ten papers regardless of which technique they study. First,
proxy metrics substitute for measured performance more often than not: FLOPs and parameter-count
reductions are reported far more consistently than on-device latency or throughput, and in
several cases ([1], [2], [4], [6], [10]) hardware mentioned in the paper is used only for training
or accuracy evaluation, not for the inference-speed claims the introduction motivates the work
with. Second, headline numbers in several abstracts describe a best-case configuration or a
single best-performing task/comparison rather than a typical or average result ([2], [3], [6]),
which can overstate the general applicability of a method if cited without checking the
underlying table. Third, extreme compression claims are vulnerable to train/test conflation,
most starkly in [7], where reported "accuracy" in the abstract is training accuracy and the
actual test accuracy is far lower — a direct instantiation of the generalization-gap risk that
compression pipelines must guard against. These three gaps — proxy-metric substitution, best-case
reporting, and train/test conflation — motivate three concrete evaluation requirements for a
compression toolkit built on this literature: benchmark on the actual target inference runtime
rather than reporting FLOPs alone, report average or distributional results rather than
best-single-configuration numbers, and validate every compression stage against a held-out test
set rather than training-time metrics.

## References

[1] W. Chae and K. Seo, "Pipeline of Pruning, Knowledge Distillation, and Quantization for Model
Compression," *Journal of Electrical Engineering & Technology*, vol. 21, pp. 2099–2105, 2026,
doi: 10.1007/s42835-025-02541-7.

[2] Z. Wang, L. Shi, Z. Mei, X. Zhao, Z. Wang, and J. Li, "Iterative knowledge distillation and
pruning for model compression in unsupervised domain adaptation," *Pattern Recognition*, vol.
164, art. 111512, 2025, doi: 10.1016/j.patcog.2025.111512.

[3] Y. He, J. Lin, Z. Liu, H. Wang, L.-J. Li, and S. Han, "AMC: AutoML for Model Compression and
Acceleration on Mobile Devices," in *Proc. European Conference on Computer Vision (ECCV)*, 2018.

[4] M. Zhou, R. Mao, S. Sripada, and X. Li, "AEMC: Aggregator-efficient model compression for
streamlining LLMs," *Neurocomputing*, vol. 672, art. 132779, 2026, doi:
10.1016/j.neucom.2026.132779.

[5] W. Chen, H. Qiu, J. Zhuang, C. Zhang, Y. Hu, Q. Lu, T. Wang, Y. Shi, M. Huang, and X. Xu,
"Quantization of Deep Neural Networks for Accurate Edge Computing," *ACM Journal on Emerging
Technologies in Computing Systems*, vol. 17, no. 4, art. 54, 2021, doi: 10.1145/3451211.

[6] R. Abdel-Salam, A. H. Abdel-Gawad, and A. G. Wassal, "VLCQ: Post-training quantization for
deep neural networks using variable length coding," *Future Generation Computer Systems*, vol.
166, art. 107654, 2025, doi: 10.1016/j.future.2024.107654.

[7] N. A. Khan and A. M. S. Rafat, "Optimizing Deep Learning Models for Resource-Constrained
Environments With Cluster-Quantized Knowledge Distillation," *Engineering Reports*, vol. 7, art.
e70187, 2025, doi: 10.1002/eng2.70187.

[8] X. Fang, L. Chen, H. Yin, X. Chen, and W. Wang, "Communication Efficient Federated Learning
With Quantization-Aware Training Design," *IEEE Transactions on Machine Learning in
Communications and Networking*, vol. 4, pp. 45–59, 2026, doi: 10.1109/TMLCN.2025.3635050.

[9] Y. Chen, C. Li, L. Gong, X. Wen, Y. Zhang, and W. Shi, "A deep neural network compression
algorithm based on knowledge transfer for edge devices," *Computer Communications*, vol. 163, pp.
186–194, 2020, doi: 10.1016/j.comcom.2020.09.016.

[10] L. Xu, J. Ren, Z. Huang, W. Zheng, and Y. Chen, "Improving Knowledge Distillation via Head
and Tail Categories," *IEEE Transactions on Circuits and Systems for Video Technology*, vol. 34,
no. 5, pp. 3465–3478, 2024, doi: 10.1109/TCSVT.2023.3325814.

---

## 📋 Paper Structure & Writing Guidelines
*(Notes from Pratiksha Relan — 13/9/2026)*

### Structure

| Section | Content |
|---|---|
| Abstract | Summary of the work |
| Keywords | Relevant terms |
| I. Introduction | Background, Motivation, Problem Statement, Objectives, Research Contribution |
| II. Literature Survey | Graph on work before & after AI (Comparative Analysis) + Survey of work done |
| III. Material & Methods | Dataset, Architecture, Algorithm, Mathematical Model |
| IV. Results & Discussion | Accuracy, Precision, Recall |
| V. Conclusion | Summary + References |

### Section-wise Word Count Targets

| Section | Target Words |
|---|---|
| Section 2 (Literature Survey) | ~1900 words |
| Section 3 (Material & Methods) | ~1500 words |
| Section 4 (Results & Discussion) | ~1200 words |
| Section 5 (Conclusion) | ~1000 words |
| Section 6 (References) | ~800 words |

### ⚠️ Key Writing Guidelines

- **Results & Discussion:** Do **NOT** explain everything — include **only expected results** (Accuracy, Precision, Recall).
- **Mathematical Model:** Keep it **short and concise** — not too detailed.
- **Literature Survey:** Must have at least **7–8 paragraphs**.
- **Architecture:** Include the **entire diagram with explanation**.
- **Algorithm:** Present as **pseudo code**.
- This structure is for the **paper to be submitted now** — keep it focused and submission-ready.
