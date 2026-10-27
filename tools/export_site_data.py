"""Write the JSON the website reads. Every number on the site comes from one of these files.

    python tools/export_site_data.py                       # everything that is cheap
    python tools/export_site_data.py --only gradual,search # selected sections (these train/run)
    python tools/export_site_data.py --vgg-json ~/Downloads/results.json   # a Kaggle run's results.json

Sections: models, hosts, align, sensitivity, quant, bn, distill, gradual, search.
Each section is measured here, on this machine (or converted from a run's own
results.json), never typed in by hand. Hand-transcribed values are the explicit
exception and carry `"provenance": {"kind": "transcribed", ...}`.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

OUT = ROOT / "frontend-next" / "src" / "data"
CKPT = ROOT / "checkpoints" / "resnet20_fp32.pt"
SECTIONS = ["notebooks", "models", "hosts", "align", "groups", "sensitivity", "quant", "bn", "distill", "gradual", "search", "verify"]


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} KB)")


def provenance(kind: str, **extra) -> dict:
    from compresskit.core.host import probe

    host = probe().as_dict()
    return {"kind": kind, "host": host["cpu"], "machine": host["machine"], "date": date.today().isoformat(), **extra}


def load():
    from compresskit.zoo.train import load_model

    model, meta = load_model(CKPT)
    return model, meta


def loaders(batch=128):
    from compresskit.zoo.data import calibration_loader, test_loader, train_loader

    return calibration_loader(n=512, batch_size=64), test_loader(batch_size=256), train_loader(batch_size=batch)


# --------------------------------------------------------------------------- models
def condense(results: dict, *, name: str, arch: str, dataset: str, schedule: str, source: str, prov: dict) -> dict:
    """Keep what the pages render; drop raw confusion matrices (keep per-class recall)."""

    def stage(m):
        cm = m.get("class_metrics") or {}
        return {
            "stage": m["stage"], "precision": m["precision"], "accuracy": m["accuracy"],
            "median_ms": m["latency"]["median_ms"], "p95_ms": m["latency"]["p95_ms"],
            "size_mb": m["size_mb"], "parameters": m["parameters"], "notes": m.get("notes", ""),
            "recall": cm.get("recall"), "support": cm.get("support"),
            "artifact": Path(m["artifact"]).name,
        }

    return {
        "name": name, "arch": arch, "dataset": dataset, "schedule": schedule, "source": source,
        "provenance": prov,
        "host": results["host"],
        "baseline": stage(results["baseline"]),
        "stages": [stage(s) for s in results["stages"]],
        "prune_stats": results.get("prune_stats"),
        "prune_history": results.get("prune_history", []),
        "quantization": results.get("quantization"),
        "totals": results["totals"],
        "warnings": results.get("warnings", []),
    }


def copy_downloads(workdir: Path, slug: str) -> list[dict]:
    """Small ONNX files that ship with the site so a visitor can open them in any ONNX viewer."""
    import shutil

    dest = ROOT / "frontend-next" / "public" / "models"
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for stage_file, label in (("baseline.onnx", "original FP32"), ("3_quantize.onnx", "compressed INT8")):
        src = workdir / stage_file
        if src.exists():
            name = f"{slug}-{'fp32' if 'baseline' in stage_file else 'int8'}.onnx"
            shutil.copy(src, dest / name)
            files.append({"label": label, "href": f"/models/{name}", "size_bytes": src.stat().st_size})
    return files


def export_models(args) -> None:
    entries = []
    demo = ROOT / "artifacts" / "demo" / "results.json"
    if demo.exists():
        entries.append(condense(
            json.loads(demo.read_text()), name="ResNet-20", arch="resnet20", dataset="CIFAR-10",
            schedule="prune 0.5 channels, 3 steps x 2 DKD fine-tune epochs, INT8",
            source=str(demo.relative_to(ROOT)),
            prov=provenance("measured", where="local run", note="Apple M5, fast INT8"),
        ))
        entries[-1]["downloads"] = copy_downloads(demo.parent, "resnet20")
    smoke = ROOT / "artifacts" / "r56smoke" / "results.json"
    if args.resnet56_json:
        entries.append(condense(
            json.loads(Path(args.resnet56_json).read_text()), name="ResNet-56", arch="resnet56",
            dataset="CIFAR-10", schedule=args.resnet56_schedule, source=Path(args.resnet56_json).name,
            prov=provenance("measured", where="results.json supplied with --resnet56-json"),
        ))
    elif smoke.exists():
        entries.append(condense(
            json.loads(smoke.read_text()), name="ResNet-56", arch="resnet56", dataset="CIFAR-10",
            schedule="prune 0.5 channels, 2 steps x 1 DKD fine-tune epoch, INT8 (short smoke run, not the full schedule)",
            source=str(smoke.relative_to(ROOT)),
            prov=provenance("measured", where="local run", note="short schedule; superseded by the full Kaggle run when supplied"),
        ))
    if args.vgg_json:
        entries.append(condense(
            json.loads(Path(args.vgg_json).read_text()), name="VGG-16-BN", arch="vgg16_bn", dataset="CIFAR-10",
            schedule="prune 0.5 channels, 5 steps x 5 DKD fine-tune epochs, INT8",
            source=Path(args.vgg_json).name,
            prov={"kind": "measured", "host": "Kaggle (Intel Xeon @ 2.00GHz)", "where": "Kaggle notebook",
                  "note": "latency is Kaggle-CPU latency, not an INT8-capable device"},
        ))
    elif (ROOT / "tools" / "transcribed_vgg16.json").exists():
        entries.append(condense(
            json.loads((ROOT / "tools" / "transcribed_vgg16.json").read_text()), name="VGG-16-BN", arch="vgg16_bn",
            dataset="CIFAR-10", schedule="prune 0.5 channels, 5 steps x 5 DKD fine-tune epochs, INT8",
            source="tools/transcribed_vgg16.json",
            prov={"kind": "transcribed", "host": "Kaggle (Intel Xeon @ 2.00GHz)", "where": "Kaggle notebook log",
                  "note": "typed in from the run's printed log; latency is Kaggle-CPU latency, not an INT8-capable device"},
        ))
    write("models", {"models": entries})


# --------------------------------------------------------------------------- notebooks
NOTEBOOKS = [
    ("compresskit-end-to-end", "End to end", "kaggle/compresskit_end_to_end.ipynb",
     "One notebook for any supported model: train a baseline if needed, prune, fine-tune, quantize, evaluate, report."),
    ("resnet56", "ResNet-56", "kaggle/resnet56_compression.ipynb",
     "ResNet-56 on CIFAR-10 with a bundled baseline checkpoint, plus a check of the ONNX file against the original."),
]


def export_notebooks(_args) -> None:
    out = []
    for slug, title, rel, blurb in NOTEBOOKS:
        nb = json.loads((ROOT / rel).read_text())
        out.append({
            "slug": slug, "title": title, "path": rel, "blurb": blurb,
            "cells": [{"type": c["cell_type"], "source": "".join(c["source"]).strip()} for c in nb["cells"]],
        })
    write("notebooks", {"notebooks": out})


# --------------------------------------------------------------------------- hosts
def export_hosts(_args) -> None:
    from compresskit.core.host import probe

    m5 = probe().as_dict()
    write("hosts", {"hosts": [
        {**m5, "provenance": provenance("measured", where="compresskit probe"), "label": "this machine"},
        {
            # Copied from the pre-flight block printed by the Kaggle runs (no results.json carries it).
            "label": "Kaggle notebook CPU", "system": "Linux", "machine": "x86_64",
            "cpu": "Intel(R) Xeon(R) CPU @ 2.00GHz", "logical_cores": 4, "performance_cores": 4,
            "features": ["avx2", "avx512f", "f16c"],
            "int8_speed": "slow",
            "int8_note": "No VNNI/AMX/DotProd/I8MM on this CPU - INT8 may be SLOWER than FP32. Benchmark before shipping it.",
            "default_precision": "fp16",
            "provenance": {"kind": "transcribed", "where": "pre-flight output of the Kaggle ResNet-20 and VGG-16 runs"},
        },
    ]})


# --------------------------------------------------------------------------- align
def export_align(_args) -> None:
    from compresskit.pruning.prune import _aligned_ratio

    rows = []
    for width in (8, 16, 24, 32, 48, 64, 96, 128, 256, 512):
        for ratio in (0.1, 0.25, 0.3, 0.5, 0.7, 0.9):
            for align in (8, 16, 32):
                eff, used = _aligned_ratio(width, ratio, align)
                rows.append({"width": width, "ratio": ratio, "align": align, "effective_ratio": eff,
                             "align_used": used, "kept": round(width * (1 - eff))})
    write("align_golden", {"source": "compresskit.pruning.prune._aligned_ratio", "rows": rows})


# --------------------------------------------------------------------------- dependency groups
def export_groups(_args) -> None:
    from compresskit.pruning.prune import _dependency_groups, classifier_layers

    model, _ = load()
    groups = _dependency_groups(model, torch.randn(1, 3, 32, 32), classifier_layers(model))
    shared = [g for g in groups if len(g) > 1]
    write("groups", {
        "model": "ResNet-20",
        "provenance": provenance("measured", where="compresskit.pruning.prune._dependency_groups (Torch-Pruning DepGraph)"),
        "groups": groups, "multi_layer_groups": shared,
    })


# --------------------------------------------------------------------------- sensitivity
def export_sensitivity(_args) -> None:
    from compresskit.sensitivity import analyze

    model, _ = load()
    calib, _, _ = loaders()
    report = analyze(model, calib, batches=4)
    write("sensitivity", {
        "model": "ResNet-20", "bits": report.bits, "batches": report.batches,
        "provenance": provenance("measured", where="compresskit.sensitivity.analyze on 4 train-split calibration batches"),
        "layers": [{"name": l.name, "kind": l.kind, "parameters": l.parameters, "mse": l.mse, "sqnr_db": l.sqnr_db}
                   for l in report.ranked()],
        "protected_at_0.5": sorted(report.prune_ratios(0.5)),
        "kept_fp32_top2": report.most_sensitive(2),
    })


# --------------------------------------------------------------------------- quantization
def export_quant(_args) -> None:
    from compresskit.sensitivity.analysis import fake_quantize

    model, _ = load()
    layer_name = "layer2.0.conv1"
    weight = dict(model.named_modules())[layer_name].weight.detach()
    q = fake_quantize(weight, bits=8, per_channel=True)
    scales = (weight.abs().amax(dim=(1, 2, 3)) / 127).clamp(min=1e-12)
    lo, hi = float(weight.min()), float(weight.max())
    bins = 61
    edges = torch.linspace(lo, hi, bins + 1)
    h_fp = torch.histc(weight.flatten(), bins=bins, min=lo, max=hi)
    h_q = torch.histc(q.flatten(), bins=bins, min=lo, max=hi)
    levels = (weight / scales.view(-1, 1, 1, 1)).round().clamp(-128, 127)
    write("quant_weights", {
        "model": "ResNet-20", "layer": layer_name, "bits": 8, "shape": list(weight.shape),
        "provenance": provenance("measured", where="fake_quantize (the same math the sensitivity pass and QAT use)"),
        "hist": {"edges": [float(e) for e in edges], "fp32": h_fp.tolist(), "int8": h_q.tolist()},
        "scales_per_output_channel": [float(s) for s in scales],
        "distinct_int_levels_used": int(levels.unique().numel()),
        "weight_mse": float(((weight - q) ** 2).mean()),
        "weight_abs_max": float(weight.abs().max()),
        "channel0": {"scale": float(scales[0]), "weights": [float(v) for v in weight[0].flatten()]},
    })


# --------------------------------------------------------------------------- bn + distillation
def _one_shot_pruned(model):
    from compresskit.pruning import PruneConfig, StructuredPruner

    pruner = StructuredPruner(model, torch.randn(1, 3, 32, 32), PruneConfig(ratio=0.5, align=16, steps=1))
    pruner.run()
    return pruner.model.eval()


def export_bn(_args) -> None:
    from compresskit.core.benchmark import accuracy_torch
    from compresskit.pruning import recalibrate_bn

    model, _ = load()
    calib, test, _ = loaders()
    pruned = _one_shot_pruned(model)
    layer = "layer1.0.bn1"
    orig_bn = dict(model.named_modules())[layer]
    stale_bn = dict(pruned.named_modules())[layer]
    stale_var, stale_mean = stale_bn.running_var.clone(), stale_bn.running_mean.clone()
    acc_stale = accuracy_torch(pruned, test)
    recal = copy.deepcopy(pruned)
    recalibrate_bn(recal, calib, batches=8)
    new_bn = dict(recal.named_modules())[layer]
    acc_recal = accuracy_torch(recal, test)

    # The pruner keeps a subset of the original channels, so each kept channel's stale
    # statistic equals one of the original's values exactly; matching them recovers the mapping.
    kept = [int((orig_bn.running_var - v).abs().argmin()) for v in stale_var]
    write("bn_recal", {
        "model": "ResNet-20", "layer": layer, "prune_ratio": 0.5,
        "provenance": provenance("measured", where="one-shot 0.5 prune, then recalibrate_bn(batches=8) on train-split calibration data"),
        "accuracy": {"baseline": accuracy_torch(model, test), "after_cut_stale_bn": acc_stale,
                     "after_bn_recalibration": acc_recal},
        "channels_before": int(orig_bn.running_var.numel()), "channels_after": int(stale_var.numel()),
        "kept_channel_indices": kept,
        "running_var": {"original": [float(v) for v in orig_bn.running_var], "stale_after_cut": [float(v) for v in stale_var],
                        "recalibrated": [float(v) for v in new_bn.running_var]},
        "running_mean": {"original": [float(v) for v in orig_bn.running_mean], "stale_after_cut": [float(v) for v in stale_mean],
                         "recalibrated": [float(v) for v in new_bn.running_mean]},
    })

    # distillation sample: teacher = original, student = the recalibrated one-shot-pruned model (no fine-tune yet)
    export_distill_from(model, recal, test)


def export_distill_from(teacher, student, test) -> None:
    from compresskit.distillation import DistillConfig
    from compresskit.distillation.distill import dkd_loss

    cfg = DistillConfig(loss="dkd", epochs=1, device="cpu")
    images, targets = next(iter(test))
    with torch.no_grad():
        t_logits, s_logits = teacher(images), student(images)
    pick = next(i for i in range(len(targets)) if t_logits[i].argmax() == targets[i] and s_logits[i].argmax() != targets[i])
    t, s, y = t_logits[pick:pick + 1], s_logits[pick:pick + 1], targets[pick:pick + 1]
    T = cfg.temperature
    mask = F.one_hot(y, t.size(1)).bool()

    def binary(p):
        tp = p[mask].unsqueeze(1).clamp(1e-7, 1 - 1e-7)
        return torch.cat([tp, 1 - tp], dim=1)

    sp, tp_ = F.softmax(s / T, 1), F.softmax(t / T, 1)
    tckd = float(F.kl_div(binary(sp).log(), binary(tp_), reduction="batchmean") * T**2)
    ms, mt = s / T - 1000.0 * mask, t / T - 1000.0 * mask
    nckd = float(F.kl_div(F.log_softmax(ms, 1), F.softmax(mt, 1), reduction="batchmean") * T**2)
    total = float(dkd_loss(s, t, y, T, cfg.alpha, cfg.beta))
    assert abs(total - (cfg.alpha * tckd + cfg.beta * nckd)) < 1e-4, (total, tckd, nckd)
    names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    write("distillation", {
        "teacher": "ResNet-20 original", "student": "one-shot 0.5-pruned + BN recalibrated, not yet fine-tuned",
        "provenance": provenance("measured", where="one real CIFAR-10 test image the teacher gets right and the student gets wrong"),
        "classes": names, "target": int(y), "temperature": T, "alpha": cfg.alpha, "beta": cfg.beta,
        "teacher_logits": t[0].tolist(), "student_logits": s[0].tolist(),
        "teacher_softmax_T1": F.softmax(t, 1)[0].tolist(), "student_softmax_T1": F.softmax(s, 1)[0].tolist(),
        "teacher_softmax_T4": tp_[0].tolist(), "student_softmax_T4": sp[0].tolist(),
        "tckd": tckd, "nckd": nckd, "dkd_total": total, "kd_total": float(
            F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1), reduction="batchmean") * T**2),
    })


def export_distill(args) -> None:
    export_bn(args)  # the sample is derived from the same pruned student; keeps the two files consistent


# --------------------------------------------------------------------------- gradual vs one-shot
def export_gradual(_args) -> None:
    from compresskit.pipeline import Pipeline

    model, _ = load()
    calib, test, train = loaders()
    res = json.loads((ROOT / "artifacts" / "demo" / "results.json").read_text())
    pipe = Pipeline(model, calib, test, train_loader=train, workdir=ROOT / "artifacts" / "oneshot_vs_gradual")
    one = pipe.prune(0.5, align=16, steps=1, fine_tune_epochs=6, loss="dkd").run(verbose=False)
    write("gradual_vs_oneshot", {
        "model": "ResNet-20", "baseline_accuracy": res["baseline"]["accuracy"],
        "total_fine_tune_epochs": 6, "prune_ratio": 0.5,
        "provenance": provenance("measured", where="same model, same 6 fine-tune epochs, DKD; gradual = artifacts/demo, one-shot = this run"),
        "gradual": {"steps": 3, "epochs_per_step": 2, "history": res["prune_history"],
                    "final_params": res["prune_stats"]["params_after"]},
        "one_shot": {"steps": 1, "epochs_per_step": 6, "history": one.prune_history,
                     "final_params": (one.prune_stats or {}).get("params_after")},
    })


# --------------------------------------------------------------------------- search
BUDGETS = [
    ("meets with quantization alone", "accuracy_drop<1%,size<0.4mb"),
    ("needs pruning and distillation", "latency<0.45ms,accuracy_drop<8%"),
    ("cannot be met", "latency<0.02ms,accuracy_drop<0.5%"),
]


def export_search(_args) -> None:
    from compresskit.search import compress

    model, _ = load()
    calib, test, train = loaders()
    runs = []
    for title, spec in BUDGETS:
        print(f"  search: {spec}")
        result = compress(model, calib, test, budget=spec, train_loader=train, workdir=ROOT / "artifacts" / "site_search",
                          distill_epochs=3, qat_epochs=2, ratio_steps=3, prune_steps=2, prune_fine_tune_epochs=1,
                          verbose=False)
        runs.append({
            "title": title, "budget": spec, "described": result.budget.describe(), "met": result.met,
            "chosen": result.chosen.label if result.chosen else None,
            "attempts": [{
                "label": a.label, "plan": a.plan, "met": a.met, "failures": a.failures,
                "accuracy": a.result.final.accuracy, "median_ms": a.result.final.latency.median_ms,
                "size_mb": a.result.final.size_mb, "precision": a.result.final.precision, "efficiency": a.score,
            } for a in result.attempts],
            "baseline": {"accuracy": result.attempts[0].result.baseline.accuracy,
                         "median_ms": result.attempts[0].result.baseline.latency.median_ms,
                         "size_mb": result.attempts[0].result.baseline.size_mb},
        })
    write("search", {"model": "ResNet-20", "provenance": provenance("measured", where="compresskit.search.compress, shortened recovery schedules"),
                     "runs": runs})


def export_verify(_args) -> None:
    """Original PyTorch model vs the compressed .onnx on the same real test images
    (the check experiments/verify_onnx.py prints), recorded so the site can show it."""
    import numpy as np

    from compresskit.core.benchmark import make_session
    from compresskit.zoo.data import test_loader

    model, _ = load()
    onnx_path = ROOT / "artifacts" / "demo" / "3_quantize.onnx"
    session = make_session(onnx_path, threads=0)
    name = session.get_inputs()[0].name
    truth, orig, comp = [], [], []
    with torch.no_grad():
        for images, targets in test_loader(batch_size=64):
            orig += model(images).argmax(1).tolist()
            comp += session.run(None, {name: images.numpy().astype(np.float32)})[0].argmax(1).tolist()
            truth += targets.tolist()
            if len(truth) >= 1000:
                break
    truth, orig, comp = (np.array(a[:1000]) for a in (truth, orig, comp))
    write("verify", {
        "model": "ResNet-20",
        "images": int(len(truth)),
        "original_accuracy": float((orig == truth).mean()),
        "compressed_accuracy": float((comp == truth).mean()),
        "agreement": float((orig == comp).mean()),
        "provenance": provenance("measured", where="experiments/verify_onnx.py logic on artifacts/demo/3_quantize.onnx, first 1000 CIFAR-10 test images"),
    })


STEPS = {"notebooks": export_notebooks, "models": export_models, "hosts": export_hosts, "align": export_align, "groups": export_groups, "sensitivity": export_sensitivity,
         "quant": export_quant, "bn": export_bn, "distill": export_distill, "gradual": export_gradual, "search": export_search, "verify": export_verify}
CHEAP = ["notebooks", "models", "hosts", "align", "groups", "sensitivity", "quant", "bn"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help="comma-separated sections (default: the cheap ones)")
    p.add_argument("--vgg-json", help="results.json from the Kaggle VGG-16 run")
    p.add_argument("--resnet56-json", help="results.json from the full ResNet-56 run")
    p.add_argument("--resnet56-schedule", default="prune 0.5 channels, 5 steps x 5 DKD fine-tune epochs, INT8")
    args = p.parse_args(argv)
    for name in (args.only.split(",") if args.only else CHEAP):
        if name not in STEPS:
            p.error(f"unknown section {name!r}; choose from {SECTIONS}")
        print(f"[{name}]")
        STEPS[name](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
