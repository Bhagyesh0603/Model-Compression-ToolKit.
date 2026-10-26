# CompressKit — End-to-End Workflow (Locked Plan)

**Stack (fixed):** GitHub (code) + Kaggle Notebook with a T4 GPU (all compute) + Mac M5 (final speed check only).
Colab is no longer used. Model files are not copied between machines while the work is running.

---

## The flow, from start to finish

```
┌──────────────────────── ONE Kaggle notebook — "Run All" ────────────────────────┐
│                                                                                  │
│  1. SETUP        git clone repo → pip install -e . → check GPU                    │
│  2. DATA         HAR dataset attached via "Add Data" (no download, no kaggle.json)│
│  3. TRAIN        FP32 baseline  ──► checkpoints/har_resnet18_fp32.pt             │
│                  (skipped automatically if the checkpoint already exists)         │
│  4. COMPRESS     prune 50% in 5 steps, 5 DKD fine-tune epochs after each step     │
│                  → BN recalibration → INT8 quantization (always last)             │
│  5. EVALUATE     accuracy measured on the final .onnx file, not the PyTorch model │
│  6. OUTPUT       /kaggle/working/  ← model.onnx + report.html + results.json      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
                                       │  download 1 folder (once, at the end)
                                       ▼
                 Mac M5:  compresskit bench model.onnx   ← real deployment latency
```

Order rule: **train → prune + fine-tune → quantize**. Quantization always comes last,
because INT8 weights can no longer be pruned or trained.

The toolkit already runs steps 4–5 as a single command (`compresskit pipeline` /
`experiments/compress_har.py`). Nothing in this plan changes the compression engine.

---

## A single config cell controls everything

```python
MODEL       = "har_resnet18"   # or "resnet20" (CIFAR-10)
TRAIN       = "auto"           # auto = train only if no checkpoint exists
PRUNE_RATIO = 0.5
STEPS       = 5
FT_EPOCHS   = 5
LOSS        = "dkd"
PRECISION   = "auto"           # int8 if the CPU supports it, else fp16
```

---

## What was built

| # | Change | Where |
|---|--------|-------|
| 1 | HAR dataset path is configurable: `--data-root`, `$COMPRESSKIT_HAR_ROOT`, or auto-found under `/kaggle/input` | `zoo/har.py` |
| 2 | HAR / pretrained ResNet-18 training from the CLI: `compresskit train --arch har_resnet18` (with `--skip-if-exists`) | `cli.py` |
| 3 | `pipeline` / `compress` work on any dataset and input size, write to `--workdir`, `--report`, `--json` | `cli.py` |
| 4 | The notebook | `kaggle/compresskit_end_to_end.ipynb` |
| 5 | Tests for the dataset registry, root resolution, and train → gradual prune → INT8 → report | `tests/test_workflow.py` |
| 6 | Colab files moved to `archive/colab/`; README documents the single flow | `archive/`, `README.md` |
| 7 | Bring your own data (`--dataset folder`) or model (`MODEL_FILE`) | `zoo/datasets.py`, notebook config |
| 8 | Detailed report: gradual pruning schedule, channels per layer, training curve, per-class accuracy | `report.py`, `pipeline.py` |
| 9 | `MODE = "manual"` or `"auto"` (budget) in the notebook | notebook |
| 10 | Budget search uses gradual pruning + fine-tuning in its recovery stage (`--steps`, `--fine-tune-epochs`) | `search.py` |
| 11 | INT8 is forced on Kaggle (`--precision int8 --allow-slow-int8`); report states that latency is not the target's | `report.py`, notebook |
| 12 | Accuracy line now reads "drop X pp" instead of a confusing signed number | `pipeline.py` |

---

## Latency, stated honestly

- **Accuracy and size** in the Kaggle report are final: they don't depend on the machine.
- **Latency** in the Kaggle report is measured on Kaggle's CPU. The deployment latency figure
  comes from running `compresskit bench` on the M5. That is one command at the end, and it
  verifies the model; it is not a pipeline step.
- If Kaggle's CPU fails the INT8 check, the notebook passes `--allow-slow-int8` so the run still
  produces the INT8 model. The toolkit's rule is unchanged: real latency is only ever claimed
  from the target hardware.

---

## How you run it (after this is built)

1. Kaggle → New Notebook → import `kaggle/compresskit_end_to_end.ipynb` from GitHub.
2. Settings: Accelerator **GPU T4**, Internet **ON**, Add Data → *Human Action Recognition*.
3. Set the config cell → **Run All**.
4. Download the output folder → on the Mac: `compresskit bench model.onnx --input-shape 1 3 224 224`.
