"""Train the FP32 baselines everything else is measured against.

The baseline is doing double duty: it is the accuracy reference *and* the
distillation teacher, so it is worth training properly once rather than
re-training a weak one per experiment.

    python experiments/train_baseline.py resnet56
    python experiments/train_baseline.py vgg16_bn --epochs 80
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit.core.benchmark import accuracy_torch  # noqa: E402
from compresskit.core.host import probe  # noqa: E402
from compresskit.data import test_loader, train_loader  # noqa: E402
from compresskit.models import ARCHITECTURES, build  # noqa: E402
from compresskit.train import TrainConfig, save_model, train  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("arch", nargs="?", default="resnet56", choices=sorted(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", help="checkpoint path (default checkpoints/<arch>_fp32.pt)")
    args = parser.parse_args(argv)

    host = probe()
    print(host.describe())
    print(f"\ntraining {args.arch} for {args.epochs} epochs on {TrainConfig().device}\n")

    tl = train_loader(batch_size=args.batch_size)
    vl = test_loader(batch_size=256)
    model = build(args.arch)
    best = 0.0
    started = time.perf_counter()

    def on_epoch(_, epoch, stats):
        nonlocal best
        best = max(best, stats.get("val_acc", 0.0))
        print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {stats['loss']:.3f}  "
              f"train {stats['train_acc'] * 100:5.2f}%  val {stats.get('val_acc', 0) * 100:5.2f}%  "
              f"lr {stats['lr']:.4f}  {stats['seconds']:.1f}s", flush=True)

    train(model, tl, TrainConfig(epochs=args.epochs, lr=args.lr), val_loader=vl, on_epoch=on_epoch)

    # Report the final model's accuracy, not the best epoch's — the final model
    # is the one that gets saved and compressed.
    final = accuracy_torch(model, vl)
    path = save_model(model, args.out or f"checkpoints/{args.arch}_fp32.pt",
                      {"arch": args.arch, "epochs": args.epochs, "accuracy": final,
                       "dataset": "cifar10", "host": host.cpu})

    print(f"\nfinal top-1 {final * 100:.2f}%  (best epoch {best * 100:.2f}%)  "
          f"in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
