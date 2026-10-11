"""The escalation logic: cheapest stage that works, and stop there."""

import pytest
import torch.nn as nn

from compresskit.core import host as host_mod
from compresskit.core.host import HostContext
from compresskit.search import _new_warnings, _warning_key, compress

FAST = dict(search_runs=4, search_accuracy_limit=64, ratio_steps=1, distill_epochs=1, verbose=False)


def _host(features: frozenset[str]) -> HostContext:
    return HostContext(system="Linux", machine="x86_64", cpu="Fake CPU", logical_cores=4,
                       performance_cores=4, features=features, versions={})


@pytest.fixture
def fast_int8_host(monkeypatch):
    """Pins the host so quantize-only produces INT8 regardless of the machine running the tests."""
    monkeypatch.setattr(host_mod, "probe", lambda: _host(frozenset({"avx512_vnni"})))


@pytest.fixture
def no_int8_host(monkeypatch):
    """Pins a host where 'auto' precision resolves to fp16, not INT8."""
    monkeypatch.setattr(host_mod, "probe", lambda: _host(frozenset({"avx2"})))


class TestEscalation:
    def test_stops_at_quantize_only_when_that_is_enough(self, model, loader, tmp_path):
        """The headline behaviour: never run pruning or fine-tuning you didn't need."""
        result = compress(model, loader, loader, budget="size<10mb",
                          workdir=tmp_path / "s", **FAST)

        assert result.met
        assert len(result.attempts) == 1
        assert result.chosen.label == "quantize-only"
        assert result.sensitivity is None, "sensitivity analysis is wasted work if we stopped early"

    def test_tries_qat_before_pruning_when_only_accuracy_is_short(self, model, loader, tmp_path):
        """Quantize-only already meets latency here (an unconstrained-latency
        budget can never fail on latency), so accuracy is the only possible
        failure — QAT recovery should be tried before pruning ever starts,
        since pruning would only make accuracy worse, not better."""
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", **FAST)

        labels = [a.label for a in result.attempts]
        assert labels[0] == "quantize-only"
        assert "qat+quantize" in labels
        assert labels.index("qat+quantize") < next(
            (i for i, l in enumerate(labels) if "prune@" in l), len(labels)
        ), "QAT should be tried before pruning, not after"

    def test_tries_mixed_precision_before_qat_when_only_accuracy_is_short(
        self, model, loader, tmp_path, fast_int8_host
    ):
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", **FAST)
        labels = [a.label for a in result.attempts]
        assert labels[:2] == ["quantize-only", "mixed-precision"]
        assert labels.index("mixed-precision") < labels.index("qat+quantize")
        mixed = result.attempts[1]
        assert len(mixed.config["keep_float"]) == 2
        # Strictly more float32 compute than quantize-only, so the exclusions
        # demonstrably took effect rather than merely being non-empty.
        assert sum(mixed.result.quantization.unquantized_ops.values()) > sum(
            result.attempts[0].result.quantization.unquantized_ops.values()
        ), "the kept layers must actually stay float32"
        assert result.sensitivity is not None

    def test_keep_float_zero_skips_mixed_precision(self, model, loader, tmp_path, fast_int8_host):
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", keep_float=0, **FAST)
        assert "mixed-precision" not in [a.label for a in result.attempts]

    def test_no_mixed_precision_when_quantize_only_is_not_int8(self, model, loader, tmp_path, no_int8_host):
        """Exclusions only act on the INT8 path. On a host that gets fp16, a
        'mixed-precision' attempt would be quantize-only relabelled."""
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", **FAST)
        assert result.attempts[0].result.quantization.precision != "int8"
        assert "mixed-precision" not in [a.label for a in result.attempts]

    def test_escalates_to_pruning_when_latency_is_short(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **FAST)

        labels = [a.label for a in result.attempts]
        assert labels[0] == "quantize-only"
        assert any("prune@" in label for label in labels), "should have tried pruning"
        assert result.sensitivity is not None, "pruning ratios must come from the sensitivity ranking"

    def test_returns_best_effort_when_the_budget_is_impossible(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **FAST)

        assert not result.met
        assert result.chosen is not None, "an impossible budget must still return the best candidate"
        assert "BEST EFFORT" in result.summary()

    def test_unconstrained_budget_is_met_by_the_cheapest_stage(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="", workdir=tmp_path / "s", **FAST)
        assert result.met and len(result.attempts) == 1

    def test_chosen_result_is_remeasured_at_full_fidelity(self, model, loader, tmp_path):
        """Search measures cheaply; the reported result must not be that estimate."""
        result = compress(model, loader, loader, budget="size<10mb",
                          workdir=tmp_path / "s", **FAST)

        assert result.chosen.result.final.latency.runs == 200
        assert result.chosen.result.baseline.latency.runs == 200

    def test_summary_explains_every_attempt(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **FAST)
        summary = result.summary()

        assert "budget:" in summary
        for attempt in result.attempts:
            assert attempt.label in summary
        assert "latency" in summary, "the shortfall should name the failing constraint"


class TestSearchReport:
    def test_renders_an_attempts_table(self, model, loader, tmp_path):
        from compresskit.report import render_search

        result = compress(model, loader, loader, budget="size<10mb",
                          workdir=tmp_path / "s", **FAST)
        path = render_search(result, tmp_path / "search.html")
        text = path.read_text()

        assert "Budget search" in text
        assert result.chosen.label in text
        assert "data:image/png;base64," in text


class TestPruneWarnings:
    def test_alignment_warnings_reach_the_search_summary(self, loader, tmp_path):
        """`Pipeline.run()` is always called with verbose=False during search
        (see the `run()` helper in search.py), so a prune stage's own
        alignment/ratio-conflict warnings never print on their own — without
        `Attempt.prune_warnings` surfacing them, a warning that the requested
        ratio wasn't actually honored would be invisible even though it
        undermines every conclusion the search draws from that attempt."""
        narrow = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        model = nn.Sequential(narrow, nn.Flatten(1), nn.Linear(16, 10))

        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **FAST)

        pruned_attempts = [a for a in result.attempts if "prune@" in a.label]
        assert pruned_attempts, "expected the search to have tried pruning"
        assert any(a.prune_warnings for a in pruned_attempts), (
            "a 16-wide layer at the default align=16 should trigger a per-layer "
            "alignment warning"
        )
        assert "!" in result.summary()

    def test_warning_key_masks_ratio_numbers_but_not_layer_names(self):
        a = "['layer1.2.bn1'] share a connection with a layer overridden to 0.225 — using 0.225"
        b = "['layer1.2.bn1'] share a connection with a layer overridden to 0.1125 — using 0.1125"
        c = "['layer3.1.bn1'] share a connection with a layer overridden to 0.225 — using 0.225"
        assert _warning_key(a) == _warning_key(b), "same issue at a different ratio is one warning"
        assert _warning_key(a) != _warning_key(c), "a different layer is a different warning"

    def test_warning_key_keeps_decimal_looking_layer_names_distinct(self):
        assert _warning_key("['0.0'] cannot be pruned at all") != _warning_key("['0.1'] cannot be pruned at all")
        assert _warning_key("{'0.0': 8} use a smaller alignment") != _warning_key("{'0.1': 8} use a smaller alignment")
        assert _warning_key("0.0: requested ratio 0.300 but") != _warning_key("0.1: requested ratio 0.300 but")
        assert _warning_key("0.0: requested ratio 0.300 but") == _warning_key("0.0: requested ratio 0.450 but")

    def test_new_warnings_returns_each_key_once(self):
        seen: set[str] = set()
        first = _new_warnings(["x at 0.5", "y"], seen)
        second = _new_warnings(["x at 0.25", "y", "z"], seen)
        assert first == ["x at 0.5", "y"]
        assert second == ["z"]

    def test_summary_prints_each_repeated_warning_once_and_counts_the_rest(self, loader, tmp_path):
        narrow = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        model = nn.Sequential(narrow, nn.Flatten(1), nn.Linear(16, 10))
        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **{**FAST, "ratio_steps": 3})

        all_warnings = [w for a in result.attempts for w in a.prune_warnings]
        distinct = {_warning_key(w) for w in all_warnings}
        assert len(all_warnings) > len(distinct), "fixture should produce repeated warnings"

        printed = [line.strip()[2:] for line in result.summary().splitlines() if line.strip().startswith("! ")]
        assert len(printed) == len(distinct)
        assert f"{len(all_warnings) - len(distinct)} repeated" in result.summary()
        # The full list is still available on each attempt: dedup is display-only.
        assert sum(len(a.prune_warnings) for a in result.attempts) == len(all_warnings)


class TestDuplicateArchitectures:
    def test_no_two_prune_attempts_measure_the_same_architecture(self, model, loader, tmp_path):
        """Alignment rounding makes neighbouring ratios collapse onto the same
        network; measuring it twice is pure waste."""
        result = compress(model, loader, loader, budget="latency<0.00001ms",
                          workdir=tmp_path / "s", **{**FAST, "ratio_steps": 4})

        pruned = [a for a in result.attempts if "prune@" in a.label and "distill" not in a.label]
        archs = [tuple(sorted((k, v[1]) for k, v in a.result.prune_stats["channels"].items()))
                 for a in pruned]
        assert len(archs) == len(set(archs)), f"re-measured a duplicate architecture: {[a.label for a in pruned]}"
        assert len(pruned) < 4, "this fixture's later ratios should collapse onto one architecture"
        assert not result.met and result.chosen is not None, "the skip must not break best-effort selection"
