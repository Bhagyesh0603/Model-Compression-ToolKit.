"""Budget search with selection on a held-out validation split.

The ResNet-20 used here is trained on 45,000 training images (`compresskit train
--holdout-val 5000`), so the 5,000 held-out images are unseen by every model in
the search. Candidates are ranked and the budget verdict is decided on that
split; the untouched test set scores only the winner. Same budgets and shortened
schedules as the site's search (tools/export_site_data.py).
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BUDGETS = [
    ("meets with quantization alone", "accuracy_drop<1%,size<0.4mb"),
    ("needs pruning and distillation", "latency<0.45ms,accuracy_drop<8%"),
    ("cannot be met", "latency<0.02ms,accuracy_drop<0.5%"),
]


def main() -> int:
    from compresskit.search import compress
    from compresskit.zoo.datasets import get_bundle
    from compresskit.zoo.train import load_model

    model, meta = load_model(ROOT / "checkpoints" / "resnet20_holdout45k.pt")
    bundle = get_bundle("cifar10", None, holdout_val=5000)
    calib, test, train = bundle.calibration_loader(n=512), bundle.test_loader(), bundle.train_loader()
    out = []
    for title, spec in BUDGETS:
        print(f"\n=== {title}: {spec}", flush=True)
        r = compress(model, calib, test, budget=spec, train_loader=train, select_loader=bundle.val_loader(),
                     workdir=ROOT / "artifacts" / "val_search", distill_epochs=3, qat_epochs=2, ratio_steps=3,
                     prune_steps=2, prune_fine_tune_epochs=1, verbose=False)
        c = r.chosen
        out.append({
            "title": title, "budget": spec, "met": r.met, "chosen": c.label if c else None,
            "validation": {"accuracy": c.result.final.accuracy, "baseline": c.result.baseline.accuracy,
                           "median_ms": c.result.final.latency.median_ms, "size_mb": c.result.final.size_mb} if c else None,
            "test": r.test_accuracy,
            "attempts": [{"label": a.label, "met": a.met, "failures": a.failures, "val_accuracy": a.result.final.accuracy,
                          "median_ms": a.result.final.latency.median_ms} for a in r.attempts],
        })
        print(json.dumps(out[-1], indent=1)[:1500], flush=True)
    (ROOT / "artifacts" / "val_search_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote artifacts/val_search_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
