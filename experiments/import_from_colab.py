"""Wrap a state_dict trained in colab/train_resnet18.ipynb into a compresskit checkpoint.

Colab exports only the state_dict (see colab/train_resnet18.ipynb) so the file
has no dependency on that notebook's runtime being importable later. This script
rebuilds the same architecture locally, loads the weights into it, and re-measures
accuracy on the local test split rather than trusting Colab's reported number —
that number came from a different DataLoader/library-version instantiation, and
a mismatch here (e.g. ~10% = chance) is the fastest signal that something about
the state_dict didn't round-trip cleanly.

    python experiments/import_from_colab.py resnet18_cifar10.pt
    python experiments/import_from_colab.py resnet18_cifar10.pt --out checkpoints/resnet18_fp32.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from compresskit.benchmark import accuracy_torch  # noqa: E402
from compresskit.data import test_loader  # noqa: E402
from compresskit.host import probe  # noqa: E402
from compresskit.models import ARCHITECTURES, build  # noqa: E402
from compresskit.train import save_model  # noqa: E402

import torch  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_dict_path", help="the .pt downloaded from Colab")
    parser.add_argument("--arch", default=None,
                        help="override the arch name stored in the file (default: read from file)")
    parser.add_argument("--out", help="checkpoint path (default checkpoints/<arch>_fp32.pt)")
    args = parser.parse_args(argv)

    payload = torch.load(args.state_dict_path, map_location="cpu", weights_only=False)
    if "state_dict" not in payload:
        parser.error(
            f"{args.state_dict_path} has no 'state_dict' key — expected the file produced by "
            "colab/train_resnet18.ipynb's save cell, not a raw torch.save(model.state_dict(), ...)"
        )

    arch = args.arch or payload.get("arch")
    if arch not in ARCHITECTURES:
        parser.error(f"unknown arch {arch!r}; choose from {sorted(ARCHITECTURES)} or pass --arch")

    model = build(arch)
    missing_unexpected = model.load_state_dict(payload["state_dict"], strict=True)
    print(f"loaded {arch} state_dict: {missing_unexpected}")

    model.eval()
    vl = test_loader(batch_size=256)
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
    if local_accuracy < 0.15:
        print("! this is close to chance-level (10% on CIFAR-10) — the weights likely did not load correctly")

    host = probe()
    path = save_model(
        model, args.out or f"checkpoints/{arch}_fp32.pt",
        {"arch": arch, "accuracy": local_accuracy, "dataset": "cifar10",
         "source": "colab", "host": host.cpu},
    )
    print(f"saved -> {path}")
    print(f"\nnow usable exactly like the local baselines, e.g.:\n"
          f"  compresskit compress --model {path} --budget \"latency<0.5ms,accuracy_drop<1%\" "
          f"--report artifacts/resnet18_report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
