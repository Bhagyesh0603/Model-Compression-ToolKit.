# Demo script — run this, don't improvise

Two acts. Act 1 is the safe, always-works baseline (~10s). Act 2 shows off
the new QAT-recovery escalation (~30s). Do NOT attempt a pruning/latency-
guided run live — it takes ~3 minutes and prints a lot of warning noise;
mention it exists instead, and show the report from a pre-run if asked.

Run both commands once before the actual demo so the exact numbers below
match what your machine prints — they'll be close but not byte-identical
(QAT involves randomness).

---

## 30-second pitch (say this first)

> "Most compression tools just shrink numbers and hope it's faster. Ours
> checks your actual CPU first — some CPUs don't even speed up with INT8,
> they just get smaller and slower — and it only does as much work as
> needed: quantize first, and escalates to retraining or pruning only if
> that alone doesn't hit the target."

---

## Act 1 — quantize-only meets the budget (~10s)

```bash
cd "/Users/bhagyeshjoshi/Developer/Model Compression ToolKit."
./.venv/bin/python -m compresskit.cli compress \
  --model checkpoints/resnet20_fp32.pt --arch resnet20 \
  --budget "latency<5ms,accuracy_drop<3%" \
  --report demo_act1.html
```

**While it runs, say:** "It's reading my CPU's capabilities right now —
that's the `PRE-FLIGHT` section — then trying the cheapest technique first."

**Point at these lines when it finishes:**
```
host      : Apple M5 ... i8mm, dotprod ...
INT8      : fast — expect a real latency win.
1_quantize   int8  acc 91.54%  lat 0.373 ms  size 0.31 MB
budget met — stopping.
```

**Say:** *"91.54% accuracy, unchanged. 0.31 MB instead of the original size.
It stopped there because the budget was already met — it never wastes time
retraining or pruning when it doesn't need to."*

Then: `open demo_act1.html` — point at the chart and cards.

---

## Act 2 — quantize-only misses, QAT recovers it (~30s)

```bash
./.venv/bin/python -m compresskit.cli compress \
  --model checkpoints/resnet20_fp32.pt --arch resnet20 \
  --budget "latency<0.5ms,accuracy_drop<0.1%" --qat-epochs 1 \
  --report demo_act2.html
```

**Say before running:** "Now let's tighten the accuracy requirement so
plain quantization isn't quite enough, and watch what it does automatically."

**Point at this transition when it finishes:**
```
[MISS] quantize-only     acc 91.60%   - accuracy drop 0.20 pp > 0.10 pp
[MET ] qat+quantize      acc 91.69%
CHOSEN: qat+quantize — budget met, stopped here.
```

**Say:** *"Quantization alone missed by 0.2 points. Instead of jumping
straight to full pruning and a 10-epoch retrain, it tried one epoch of
quantization-aware training first — simulating the rounding error during
training so the weights compensate for it — and that alone not only met
the budget, it actually improved accuracy slightly. It only escalates
further than this if one epoch isn't enough."*

---

## Judge Q&A cheat sheet

- **"Why not just use PyTorch's built-in quantization?"** → "It doesn't
  check whether the CPU actually benefits — on older CPUs, INT8 can be
  slower. Ours probes the hardware first."
- **"What if QAT alone isn't enough either?"** → "It escalates further:
  structured pruning next, guided by two signals — which layers are most
  sensitive to quantization, and which layers actually cost real measured
  latency, not just FLOPs — then distillation as a last resort. Never runs
  a stage it doesn't need."
- **"How do you know accuracy didn't just look fine by luck?"** → "Accuracy
  is measured on the exported file that would actually ship, not the
  original model in memory — export bugs show up in the number instead of
  hiding."
- **"What's novel here vs. existing tools?"** → Nobody else gates the
  precision decision on the real CPU, and no maintained tool combines
  prune/quantize/distill/QAT with automatic escalation anymore — the bigger
  frameworks (NNCF, TF-MOT, NNI) have all been dropping features or gone
  unmaintained.
- **"What's not done yet?"** → Two research-grade extensions are explicitly
  future work: AdaRound-style advanced calibration, and RepVGG/MobileOne
  structural reparameterization (which needs models retrained from scratch,
  not a compression-time add-on).
