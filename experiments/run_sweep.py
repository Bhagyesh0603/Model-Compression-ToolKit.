"""Ratio x precision sweep — the Pareto front, measured rather than argued.

Runs the full pipeline once per (pruning ratio, precision) pair and writes a CSV
plus an HTML report ranking every configuration by efficiency score. This is what
turns "should I prune 30% or 50%?" into a table you can read.

    python experiments/run_sweep.py --model checkpoints/resnet56_fp32.pt
    python experiments/run_sweep.py --model ckpt.pt --ratios 0 0.3 0.5 --distill 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit.core.host import probe  # noqa: E402
from compresskit.data import calibration_loader, test_loader, train_loader  # noqa: E402
from compresskit.pipeline import Pipeline  # noqa: E402
from compresskit.report import render_sweep  # noqa: E402
from compresskit.sensitivity import analyze  # noqa: E402
from compresskit.train import load_model  # noqa: E402

FIELDS = ["config", "ratio", "precision", "distill_epochs", "top1", "drop_pp",
          "median_ms", "p95_ms", "speedup", "size_mb", "params", "efficiency"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="checkpoint from train_baseline.py")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7])
    parser.add_argument("--precisions", nargs="+", default=["fp32", "int8"])
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--distill", type=int, default=0, help="distillation epochs per config")
    parser.add_argument("--loss", default="dkd", choices=["kd", "dkd"])
    parser.add_argument("--outdir", default="artifacts/sweep")
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(probe().describe(), "\n")
    model, meta = load_model(args.model)
    print(f"baseline: {meta.get('arch', '?')} @ {meta.get('accuracy', 0) * 100:.2f}% top-1\n")

    calib = calibration_loader(n=512, batch_size=64)
    test = test_loader(batch_size=256)
    train = train_loader(batch_size=128) if args.distill else None

    print("per-layer sensitivity (drives the per-layer pruning ratios):")
    sensitivity = analyze(model, calib, batches=4)
    print(sensitivity.summary(top=5), "\n")

    baseline = None
    rows, csv_rows = [], []
    started = time.perf_counter()

    for ratio in args.ratios:
        for precision in args.precisions:
            label = f"prune{ratio:.2f}_{precision}" + (f"_kd{args.distill}" if args.distill else "")
            print(f"--- {label} ---", flush=True)

            pipe = Pipeline(model, calib, test, train_loader=train,
                            workdir=outdir / label, warmup=50, runs=200)
            if ratio > 0:
                pipe.prune(ratio=ratio, align=args.align,
                           ratio_overrides=sensitivity.prune_ratios(ratio)).recover()
            if args.distill:
                pipe.distill(epochs=args.distill, loss=args.loss)
            pipe.quantize(precision=precision)

            result = pipe.run(verbose=False)
            baseline = baseline or result.baseline
            final = result.final

            print(f"    {final.summary()}")
            rows.append({"label": label, "measurement": final})
            csv_rows.append({
                "config": label, "ratio": ratio, "precision": final.precision,
                "distill_epochs": args.distill,
                "top1": round(final.accuracy * 100, 3),
                "drop_pp": round((baseline.accuracy - final.accuracy) * 100, 3),
                "median_ms": round(final.latency.median_ms, 4),
                "p95_ms": round(final.latency.p95_ms, 4),
                "speedup": round(baseline.latency.median_ms / final.latency.median_ms, 3),
                "size_mb": round(final.size_mb, 4),
                "params": final.parameters,
                "efficiency": round(result.efficiency_score, 4),
            })

    csv_path = outdir / "sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

    html_path = render_sweep(baseline, rows, outdir / "sweep.html",
                             title=f"{meta.get('arch', 'model')} — ratio x precision sweep")

    best = max(csv_rows, key=lambda r: r["efficiency"])
    print(f"\n{len(csv_rows)} configs in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"best by efficiency: {best['config']} — {best['top1']:.2f}% top-1, "
          f"{best['speedup']:.2f}x faster, {best['size_mb']:.2f} MB")
    print(f"csv    -> {csv_path}")
    print(f"report -> {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
