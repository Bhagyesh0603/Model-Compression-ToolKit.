"""Wrap a state_dict trained in colab/train_resnet18_har.ipynb into a compresskit checkpoint.

Same philosophy as import_from_colab.py: Colab exports only the state_dict, so
this rebuilds the exact architecture locally, loads the weights, and
re-measures accuracy on the LOCAL test split rather than trusting Colab's
reported number. A mismatch here (e.g. ~6.7% = chance for 15 classes) is the
fastest signal that the state_dict didn't round-trip cleanly, or that the two
class-to-index orderings don't actually match.

Requires the dataset already downloaded locally (see README step in
compresskit/har.py's error message, or just run):

    kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset \\
        -p data/har --unzip

Usage:
    python experiments/import_har_checkpoint.py har_resnet18.pt
    python experiments/import_har_checkpoint.py har_resnet18.pt --out checkpoints/har_resnet18_fp32.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit.benchmark import accuracy_torch  # noqa: E402
from compresskit.har import build_har_resnet18, class_names, test_loader  # noqa: E402
from compresskit.host import probe  # noqa: E402
from compresskit.train import save_model  # noqa: E402

import torch  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_dict_path", help="the .pt downloaded from Colab")
    parser.add_argument("--out", default="checkpoints/har_resnet18_fp32.pt",
                        help="checkpoint path (default checkpoints/har_resnet18_fp32.pt)")
    args = parser.parse_args(argv)

    payload = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        parser.error(
            f"{args.state_dict_path} has no 'state_dict' key — expected the file produced by "
            "colab/train_resnet18_har.ipynb's save cell, not a raw torch.save(model.state_dict(), ...)"
        )

    num_classes = payload.get("num_classes", 15)
    model = build_har_resnet18(num_classes=num_classes, pretrained=False)
    missing_unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    print(f"loaded har_resnet18 state_dict: {missing_unexpected}")

    remote_classes = payload.get("class_names")
    local_classes = class_names()
    if remote_classes is not None and remote_classes != local_classes:
        print("! class order mismatch between Colab and the local dataset:")
        print(f"    colab: {remote_classes}")
        print(f"    local: {local_classes}")
        print("  predictions will be silently wrong (transposed labels) — do not proceed "
              "until these match, e.g. by re-downloading the same dataset version locally.")
        return 1

    model.eval()
    vl = test_loader(batch_size=64)
    local_accuracy = accuracy_torch(model, vl)
    colab_accuracy = payload.get("accuracy")

    print(f"accuracy measured locally : {local_accuracy * 100:.2f}%")
    if colab_accuracy is not None:
        print(f"accuracy reported by Colab: {colab_accuracy * 100:.2f}%")
        if abs(local_accuracy - colab_accuracy) > 0.02:
            print(
                "! more than 2pp apart from what Colab reported — check the state_dict actually "
                "round-tripped (e.g. wrong normalisation stats, wrong test split) before trusting this"
            )
    chance = 1.0 / num_classes
    if local_accuracy < chance + 0.05:
        print(f"! this is close to chance-level ({chance*100:.1f}% for {num_classes} classes) — "
              "the weights likely did not load correctly")

    host = probe()
    path = save_model(
        model, args.out,
        {"arch": "har_resnet18", "accuracy": local_accuracy, "dataset": "har-15class",
         "num_classes": num_classes, "class_names": local_classes,
         "source": "colab", "host": host.cpu},
    )
    print(f"saved -> {path}")
    print(f"\nnow usable exactly like the CIFAR baselines, e.g.:\n"
          f"  compresskit compress --model {path} "
          f"--budget \"latency<5ms,accuracy_drop<1%\" "
          f"--report artifacts/har_resnet18_report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
