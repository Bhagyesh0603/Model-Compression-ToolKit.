"""Large-model machinery: AMP no-ops, batch fitting, accumulation, upscaling, bench budget.

CUDA-only paths cannot run in CI, so they are exercised through monkeypatched stand-ins
that reproduce the failure they guard against (an OOM, a slow session).
"""

import contextlib
import copy
import importlib
import time

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from compresskit.core import amp, memory
from compresskit.distillation import DistillConfig, distill
from compresskit.zoo.datasets import UpscaledLoader, loader_kwargs

# The packages re-export functions under these names, so import the modules explicitly.
bench = importlib.import_module("compresskit.core.benchmark")
distill_mod = importlib.import_module("compresskit.distillation.distill")


# --- AMP helpers are inert off-CUDA -------------------------------------------------

def test_amp_helpers_are_noops_on_cpu():
    scaler = amp.Scaler("cpu")
    assert not scaler.active and not amp.amp_enabled("cpu") and not amp.amp_enabled("mps")
    with amp.autocast("cpu"):
        assert torch.ones(2).dtype == torch.float32
    x = torch.randn(2, 3, 4, 4)
    assert amp.prepare_batch(x, "cpu").is_contiguous()
    layer = nn.Linear(2, 2)
    param = layer.weight.detach().clone()
    out = layer(torch.randn(3, 2)).sum()
    scaler.backward(out)
    opt = torch.optim.SGD(layer.parameters(), lr=0.1)
    scaler.step(opt)
    assert not torch.equal(param, layer.weight)


def test_amp_can_be_disabled_even_on_cuda_names():
    assert not amp.amp_enabled("cuda", requested=False)


# --- batch fitting ------------------------------------------------------------------

class _Oomer(nn.Module):
    """Fails like a GPU does once the batch is too large."""

    def __init__(self, limit):
        super().__init__()
        self.limit, self.fc, self.seen = limit, nn.Linear(4, 3), []

    def forward(self, x):
        self.seen.append(len(x))
        if len(x) > self.limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return self.fc(x.flatten(1))


@pytest.fixture
def fake_cuda(monkeypatch):
    monkeypatch.setattr(memory, "prepare_model", lambda m, d: m)
    monkeypatch.setattr(memory, "prepare_batch", lambda x, d: x)
    monkeypatch.setattr(memory, "autocast", lambda d, e=True: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_fit_batch_size_halves_until_it_fits(fake_cuda):
    model = _Oomer(limit=10)
    assert memory.fit_batch_size(model, (1, 4), start=64, device="cuda") == 8
    assert model.seen == [64, 32, 16, 8]


def test_fit_batch_size_leaves_a_batch_that_fits_alone(fake_cuda):
    assert memory.fit_batch_size(_Oomer(limit=100), (1, 4), start=32, device="cuda") == 32


def test_fit_batch_size_reraises_when_even_one_sample_fails(fake_cuda):
    with pytest.raises(RuntimeError, match="out of memory"):
        memory.fit_batch_size(_Oomer(limit=0), (1, 4), start=4, device="cuda")


def test_fit_batch_size_does_not_swallow_other_errors(fake_cuda):
    class Broken(nn.Module):
        def forward(self, x):
            raise ValueError("shape mismatch")

    with pytest.raises(ValueError):
        memory.fit_batch_size(Broken(), (1, 4), start=8, device="cuda")


def test_fit_batch_size_is_inert_off_cuda():
    model = _Oomer(limit=0)
    assert memory.fit_batch_size(model, (1, 4), start=64, device="cpu") == 64
    assert model.seen == []


@pytest.mark.parametrize("requested,fitted,expected", [(64, 64, (64, 1)), (64, 16, (16, 4)),
                                                       (64, 24, (24, 3)), (64, 100, (64, 1)), (1, 1, (1, 1))])
def test_split_batch_keeps_the_effective_batch(requested, fitted, expected):
    micro, accum = memory.split_batch(requested, fitted)
    assert (micro, accum) == expected
    assert micro * accum >= requested


# --- gradient accumulation ----------------------------------------------------------

def test_distill_accumulation_matches_one_big_batch(monkeypatch):
    # Constant LR: the cosine schedule is a function of the micro-batch index, which
    # is exactly what differs between the two runs and is not what is under test.
    monkeypatch.setattr(distill_mod, "cosine_lr", lambda base, *a, **k: base)
    torch.manual_seed(0)
    student = nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    teacher = nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    x, y = torch.randn(8, 3, 2, 2), torch.randint(0, 5, (8,))

    def run(batch, accum):
        model = copy.deepcopy(student)
        loader = DataLoader(TensorDataset(x, y), batch_size=batch)
        distill(model, copy.deepcopy(teacher), loader,
                DistillConfig(epochs=1, lr=0.1, loss="kd", device="cpu", warmup_epochs=0,
                              weight_decay=0.0, accum_steps=accum))
        return model

    one, two = run(8, 1), run(4, 2)
    for a, b in zip(one.parameters(), two.parameters()):
        assert torch.allclose(a, b, atol=1e-6)


def test_distill_rejects_bad_accum():
    with pytest.raises(ValueError):
        DistillConfig(accum_steps=0, device="cpu")


def test_distill_keep_on_device_leaves_student_where_it_trained():
    student, teacher = nn.Sequential(nn.Flatten(), nn.Linear(12, 5)), nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 2, 2), torch.randint(0, 5, (4,))), batch_size=4)
    out = distill(student, teacher, loader, DistillConfig(epochs=1, device="cpu", keep_on_device=True))
    assert out is student and not student.training


# --- upscaling ----------------------------------------------------------------------

def test_upscaled_loader_matches_the_per_sample_pil_path():
    from PIL import Image
    from torchvision import transforms

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)

    norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    old = transforms.Compose([transforms.Resize(224), *norm])(Image.fromarray(image))
    small = transforms.Compose(norm)(Image.fromarray(image))
    loader = UpscaledLoader(DataLoader(TensorDataset(small[None], torch.zeros(1, dtype=torch.long)), batch_size=1), 224)
    new = next(iter(loader))[0][0]

    assert new.shape == old.shape == (3, 224, 224)
    assert (new - old).abs().max() < 2e-2     # PIL rounds to uint8 before normalising
    assert (new - old).abs().mean() < 5e-3


def test_upscaled_loader_exposes_what_pipeline_reads():
    base = DataLoader(TensorDataset(torch.randn(6, 3, 8, 8), torch.zeros(6, dtype=torch.long)), batch_size=2)
    loader = UpscaledLoader(base, 16)
    assert len(loader) == 3 and loader.batch_size == 2 and len(loader.dataset) == 6
    images, targets = next(iter(loader))
    assert images.shape == (2, 3, 16, 16) and targets.shape == (2,)


def test_loader_kwargs_workers():
    assert "persistent_workers" not in loader_kwargs(0)
    assert loader_kwargs(0)["num_workers"] == 0
    assert loader_kwargs(3)["num_workers"] == 3
    assert 1 <= loader_kwargs(None)["num_workers"] <= 4


# --- benchmark budget ---------------------------------------------------------------

class _SlowSession:
    calls = 0

    class _In:
        name = "x"

    def get_inputs(self):
        return [self._In()]

    def run(self, *_):
        type(self).calls += 1
        time.sleep(0.05)


def test_benchmark_shrinks_to_its_time_budget(monkeypatch):
    _SlowSession.calls = 0
    monkeypatch.setattr(bench, "make_session", lambda *a, **k: _SlowSession())
    start = time.perf_counter()
    latency = bench.benchmark("unused.onnx", (1, 3, 8, 8), warmup=50, runs=200, budget_s=1.0)
    assert time.perf_counter() - start < 4.0           # unbudgeted: 250 x 50 ms = 12.5 s
    assert latency.warmup == bench.MIN_WARMUP_RUNS
    assert bench.MIN_TIMED_RUNS <= latency.runs < 200
    assert 40 < latency.median_ms < 200


def test_benchmark_keeps_full_runs_for_fast_models(monkeypatch):
    class Fast(_SlowSession):
        def run(self, *_):
            pass

    monkeypatch.setattr(bench, "make_session", lambda *a, **k: Fast())
    latency = bench.benchmark("unused.onnx", (1, 3, 8, 8), warmup=5, runs=30, budget_s=20.0)
    assert (latency.warmup, latency.runs) == (5, 30)


# --- pipeline ----------------------------------------------------------------------

def test_prune_writes_a_checkpoint_and_history_per_step(model, loader, tmp_path):
    from compresskit.pipeline import Pipeline
    from compresskit.zoo.train import load_model

    pipe = Pipeline(model, loader, loader, train_loader=loader, workdir=tmp_path, warmup=1, runs=3,
                    device="cpu").prune(ratio=0.5, align=1, steps=2, fine_tune_epochs=1)
    result = pipe.run(verbose=False)

    for step in (1, 2):
        assert (tmp_path / f"prune_step{step}.pt").exists()
    reloaded, meta = load_model(tmp_path / "prune_step2.pt")
    assert meta["prune_step"] == 2
    import json
    history = json.loads((tmp_path / "prune_history.json").read_text())
    assert [row["step"] for row in history] == [1, 2] == [row["step"] for row in result.prune_history]


def test_recover_notes_when_a_model_has_no_batchnorm(tmp_path):
    from compresskit.pipeline import Pipeline

    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 10))
    data = DataLoader(TensorDataset(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))), batch_size=4)
    result = Pipeline(net, data, data, workdir=tmp_path, warmup=1, runs=3, device="cpu").recover().run(verbose=False)
    assert "no BatchNorm" in result.stages[0].notes


def test_fit_batch_size_leaves_batchnorm_statistics_alone(fake_cuda):
    # The CLI probes with the real model standing in as its own teacher; forward passes
    # on random noise in train mode would otherwise overwrite the BN running stats and
    # corrupt the baseline that gets measured afterwards.
    net = nn.Sequential(nn.Flatten(), nn.Linear(12, 8), nn.BatchNorm1d(8), nn.Linear(8, 3))
    net.eval()
    net[2].running_mean.fill_(0.7)
    net[2].running_var.fill_(2.5)
    before = {k: v.clone() for k, v in net.state_dict().items()}

    memory.fit_batch_size(net, (1, 3, 2, 2), start=16, device="cuda", teacher=net)

    for key, value in net.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert not net.training                         # mode restored too


def test_fit_batch_size_probes_the_student_in_train_mode(fake_cuda):
    modes = []

    class Spy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 3)

        def forward(self, x):
            modes.append(self.training)
            return self.fc(x.flatten(1))

    net = Spy()
    memory.fit_batch_size(net, (1, 4), start=4, device="cuda", teacher=net)
    assert modes[-1] is True                        # the grad-carrying pass is a training pass


def test_only_the_train_loader_gets_persistent_workers():
    assert "persistent_workers" not in loader_kwargs(3)
    assert loader_kwargs(3, persistent=True)["persistent_workers"] is True
    assert "persistent_workers" not in loader_kwargs(0, persistent=True)
