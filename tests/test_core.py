"""Unit tests for the pieces that are easy to get subtly wrong."""

import pytest
import torch
import torch.nn as nn

from compresskit.core.benchmark import Latency, Measurement, benchmark, compression_score, efficiency_score
from compresskit.core.export import layer_node_map, to_onnx, torch_name_from_onnx_node
from compresskit.core.host import probe
from compresskit.distillation import DistillConfig, cosine_lr, dkd_loss, kd_loss
from compresskit.models import ARCHITECTURES, build
from compresskit.pruning import PruneConfig, classifier_layers, prune, recalibrate_bn
from compresskit.search import Budget
from compresskit.sensitivity import analyze, fake_quantize


class TestHost:
    def test_probe_is_populated_and_cached(self):
        host = probe()
        assert host.logical_cores >= 1
        assert host.recommended_precision in ("int8", "fp16")
        assert host.int8_speed in ("fast", "slow", "unknown")
        assert probe() is host

    def test_int8_verdict_follows_features(self):
        host = probe()
        # The whole point of the probe: INT8 is only "fast" when a fused kernel exists.
        assert (host.int8_speed == "fast") == bool(host.int8_features)
        assert (host.recommended_precision == "int8") == (host.int8_speed == "fast")

    def test_reduce_range_only_on_x86_without_vnni(self):
        host = probe()
        if not host.machine.lower().startswith(("x86", "amd64", "i386")):
            assert host.reduce_range is False


class TestBudget:
    @pytest.mark.parametrize(
        "spec,field,expected",
        [
            ("latency<50ms", "max_latency_ms", 50.0),
            ("latency<1.5s", "max_latency_ms", 1500.0),
            ("accuracy_drop<1%", "max_accuracy_drop_pp", 1.0),
            ("accuracy_drop<2pp", "max_accuracy_drop_pp", 2.0),
            ("accuracy>90%", "min_accuracy", 0.9),
            ("size<2mb", "max_size_mb", 2.0),
            ("size<1gb", "max_size_mb", 1024.0),
        ],
    )
    def test_parses_units(self, spec, field, expected):
        assert getattr(Budget.parse(spec), field) == pytest.approx(expected)

    def test_parses_compound_spec(self):
        budget = Budget.parse("latency<1ms, accuracy_drop<1%, size<0.5mb")
        assert budget.max_latency_ms == 1.0
        assert budget.max_accuracy_drop_pp == 1.0
        assert budget.max_size_mb == 0.5

    def test_empty_spec_is_unconstrained(self):
        assert Budget.parse("").is_empty

    def test_rejects_unparseable_spec(self):
        # Silently returning an empty budget would make everything "pass".
        with pytest.raises(ValueError, match="could not parse"):
            Budget.parse("make it fast please")

    @pytest.mark.parametrize("spec", ["latency>50ms", "size>2mb", "accuracy_drop>1%", "accuracy<90%"])
    def test_rejects_inverted_operators(self, spec):
        """'latency>50ms' silently parsed as a cap would invert the budget."""
        with pytest.raises(ValueError, match="takes"):
            Budget.parse(spec)

    def test_failures_name_the_shortfall(self):
        def m(accuracy, latency_ms, size):
            return Measurement("s", "int8", accuracy, Latency(latency_ms, latency_ms, latency_ms, 0, 1, 1, 1, 1),
                               size, 100, "x.onnx")

        baseline = m(0.90, 10.0, 4.0)
        budget = Budget.parse("latency<5ms,accuracy_drop<1%")

        assert budget.met_by(baseline, m(0.895, 4.0, 1.0))
        assert budget.failures(baseline, m(0.80, 4.0, 1.0)) == ["accuracy drop 10.00 pp > 1.00 pp"]
        assert budget.failures(baseline, m(0.895, 9.0, 1.0)) == ["latency 9.000 ms > 5.000 ms"]
        assert len(budget.failures(baseline, m(0.50, 9.0, 1.0))) == 2

    def test_failure_kinds_are_distinguished(self):
        """The search escalates differently for slow vs inaccurate — these must not blur."""
        def m(accuracy, latency_ms):
            return Measurement("s", "int8", accuracy, Latency(latency_ms, latency_ms, latency_ms, 0, 1, 1, 1, 1),
                               1.0, 100, "x.onnx")

        baseline = m(0.90, 10.0)
        budget = Budget.parse("latency<5ms,accuracy_drop<1%")

        too_slow = m(0.90, 9.0)
        assert budget.latency_failed(baseline, too_slow) and not budget.accuracy_failed(baseline, too_slow)

        inaccurate = m(0.50, 1.0)
        assert budget.accuracy_failed(baseline, inaccurate) and not budget.latency_failed(baseline, inaccurate)


class TestScores:
    def _m(self, accuracy, latency_ms, size):
        return Measurement("s", "fp32", accuracy, Latency(latency_ms, latency_ms, latency_ms, 0, 1, 1, 1, 1),
                           size, 100, "x.onnx")

    def test_compression_score_averages_size_and_speed(self):
        baseline = self._m(0.9, 10.0, 4.0)
        # half the size, half the latency -> 0.5 on both axes
        assert compression_score(baseline, self._m(0.9, 5.0, 2.0)) == pytest.approx(0.5)

    def test_no_compression_scores_zero(self):
        baseline = self._m(0.9, 10.0, 4.0)
        assert compression_score(baseline, self._m(0.9, 10.0, 4.0)) == pytest.approx(0.0)

    def test_efficiency_punishes_either_extreme(self):
        """Harmonic mean is the point — one great axis must not carry a terrible one."""
        baseline = self._m(0.9, 10.0, 4.0)
        balanced = efficiency_score(baseline, self._m(0.88, 5.0, 2.0))
        no_compression = efficiency_score(baseline, self._m(0.90, 10.0, 4.0))
        destroyed_accuracy = efficiency_score(baseline, self._m(0.001, 1.0, 0.1))

        assert no_compression == 0.0
        assert destroyed_accuracy < 0.01
        assert balanced > destroyed_accuracy


class TestNameMapping:
    @pytest.mark.parametrize(
        "node,expected",
        [
            ("/conv1/Conv", "conv1"),
            ("/layer1/layer1.0/conv1/Conv", "layer1.0.conv1"),
            ("/layer2/layer2.0/downsample/downsample.0/Conv", "layer2.0.downsample.0"),
            ("/fc/Gemm", "fc"),
        ],
    )
    def test_recovers_torch_path_from_onnx_scope(self, node, expected):
        assert torch_name_from_onnx_node(node) == expected

    @pytest.mark.parametrize("node", ["/Conv", "Conv_0", "MatMul"])
    def test_unscoped_names_map_to_nothing(self, node):
        """A functional op owns no module — naming it after its op type would
        invent a layer that cannot be overridden or excluded."""
        assert torch_name_from_onnx_node(node) == ""

    def test_every_layer_maps_on_a_real_export(self, model, tmp_path):
        path = to_onnx(model, tmp_path / "m.onnx")
        mapping = layer_node_map(path)
        expected = {n for n, m in model.named_modules() if isinstance(m, (nn.Conv2d, nn.Linear))}
        # A missed layer means a sensitivity override silently does nothing.
        assert expected == set(mapping)


class TestFakeQuantize:
    def test_stays_within_int8_grid(self):
        torch.manual_seed(0)
        weight = torch.randn(8, 4, 3, 3)
        quantized = fake_quantize(weight, bits=8)
        scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True) / 127
        levels = (quantized / scale).round()
        assert levels.abs().max() <= 128

    def test_per_channel_beats_per_tensor_on_uneven_channels(self):
        """One dominant filter is exactly what per-tensor scaling gets wrong."""
        torch.manual_seed(0)
        weight = torch.randn(4, 4, 3, 3) * 0.01
        weight[0] *= 500.0  # one huge filter sets the per-tensor scale for everyone

        per_channel_error = (weight - fake_quantize(weight, per_channel=True)).abs().mean()
        per_tensor_error = (weight - fake_quantize(weight, per_channel=False)).abs().mean()
        assert per_channel_error < per_tensor_error

    def test_zero_weights_do_not_divide_by_zero(self):
        assert torch.isfinite(fake_quantize(torch.zeros(4, 2, 3, 3))).all()


class TestDistillLosses:
    def test_kd_is_zero_when_student_matches_teacher(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 10)
        assert kd_loss(logits, logits, 4.0).item() == pytest.approx(0.0, abs=1e-6)

    def test_kd_is_positive_when_they_disagree(self):
        torch.manual_seed(0)
        assert kd_loss(torch.randn(8, 10), torch.randn(8, 10), 4.0).item() > 0

    def test_dkd_is_zero_when_student_matches_teacher(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        assert dkd_loss(logits, logits, targets, 4.0, 1.0, 8.0).item() == pytest.approx(0.0, abs=1e-5)

    def test_dkd_components_are_weighted_independently(self):
        """Decoupling is the whole idea — beta must move the loss on its own."""
        torch.manual_seed(0)
        student, teacher = torch.randn(8, 10), torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))
        low = dkd_loss(student, teacher, targets, 4.0, 1.0, 1.0).item()
        high = dkd_loss(student, teacher, targets, 4.0, 1.0, 8.0).item()
        assert high > low

    def test_short_run_is_not_all_warmup(self):
        """epochs=1 with warmup=1 would ramp the LR and never actually train."""
        config = DistillConfig(epochs=1, warmup_epochs=1)
        assert config.warmup_epochs == 0
        assert cosine_lr(config.lr, 0.0, config.epochs, config.warmup_epochs) == pytest.approx(config.lr)

    def test_dkd_survives_a_saturated_student(self):
        """A confident student drives P(target) to exactly 1.0 in fp32; without a
        clamp that is log(0) = -inf, and the run reports chance accuracy."""
        targets = torch.zeros(4, dtype=torch.long)
        saturated = torch.full((4, 10), -1e4)
        saturated[:, 0] = 1e4

        loss = dkd_loss(saturated, torch.randn(4, 10), targets, 4.0, 1.0, 8.0)
        assert torch.isfinite(loss), "DKD produced inf/NaN on a saturated student"

    def test_dkd_gradients_stay_finite_when_saturated(self):
        targets = torch.zeros(4, dtype=torch.long)
        student = torch.full((4, 10), -1e4).requires_grad_(True)
        with torch.no_grad():
            student[:, 0] = 1e4

        dkd_loss(student, torch.randn(4, 10), targets, 4.0, 1.0, 8.0).backward()
        assert torch.isfinite(student.grad).all()

    def test_alpha_defaults_per_loss(self):
        assert DistillConfig(loss="kd").alpha == 0.3
        assert DistillConfig(loss="dkd").alpha == 1.0
        assert DistillConfig(loss="dkd", alpha=0.5).alpha == 0.5


class TinyResidualNet(nn.Module):
    """Narrow (16-channel) main path with a projecting shortcut — small enough
    to run in a unit test, but shaped like a CIFAR-ResNet stage: `conv2` and
    `downsample` both feed the same addition, so Torch-Pruning ties them into
    one dependency group. That is the shape that exposes the SIMD-alignment
    starvation and cross-group ratio-override conflicts real ResNets hit."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.downsample = nn.Conv2d(16, 16, 1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.stem(x)
        identity = self.downsample(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.relu(out + identity)
        return self.classifier(self.pool(out).flatten(1))


class TestPrune:
    def test_respects_simd_alignment(self, model, example):
        pruned, stats = prune(model, example, ratio=0.5, align=16)
        for name, (_, after) in stats.channels.items():
            if name != "classifier":
                assert after % 16 == 0, f"{name} width {after} is not 16-aligned"

    def test_narrow_layers_are_not_silently_left_unpruned(self):
        """A 16-channel layer at ratio=0.5, align=16 floors the kept count to
        0, which Torch-Pruning rejects and skips — leaving the layer at 100%
        of its channels with no error. This must not happen silently: the
        wrapper should tighten alignment for that layer until pruning
        actually takes effect, and say so."""
        net = TinyResidualNet().eval()
        example = torch.randn(1, 3, 32, 32)
        pruned, stats = prune(net, example, ratio=0.5, align=16)

        for name, (before, after) in stats.channels.items():
            if name == "classifier":
                continue
            assert after < before, f"{name} was left completely unpruned ({before} -> {after})"
        assert stats.align_effective < stats.align
        assert any("SIMD alignment tightened per layer" in w for w in stats.warnings)
        assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)

    def test_narrow_layer_relaxation_does_not_weaken_wide_layers(self):
        """A stem this narrow only needs a weaker alignment for itself — the
        rest of the network must keep the full requested alignment rather
        than everyone being dragged down to whatever the narrowest layer
        needs (the old, cruder fix)."""
        net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        classifier = nn.Linear(128, 10)
        full = nn.Sequential(net, nn.Flatten(1), classifier)
        example = torch.randn(1, 3, 32, 32)

        _, stats = prune(full, example, ratio=0.5, align=16)
        narrow, wide = stats.channels["0.0"], stats.channels["0.3"]
        assert narrow[1] % 16 != 0 or narrow[1] < narrow[0], "the 16-wide stem should have been tightened"
        assert wide[1] % 16 == 0, "the 128-wide layer needed no relaxation and should keep full alignment"
        assert wide[1] < wide[0]

    def test_single_override_on_one_group_member_does_not_desync_the_group(self):
        """`conv2` and `downsample` share one Torch-Pruning group (see
        TinyResidualNet). Overriding only `conv2` must not leave it out of
        step with `downsample` once alignment rounds each independently —
        that reintroduces the exact silent "Torch-Pruning applies whichever
        value it reaches first" bug the group-conflict handling above exists
        to prevent, just self-inflicted by the alignment step instead of by
        the caller's ratio_overrides."""
        net = TinyResidualNet().eval()
        example = torch.randn(1, 3, 32, 32)
        _, stats = prune(net, example, ratio=0.5, align=16, ratio_overrides={"conv2": 0.1})

        assert stats.channels["conv2"] == stats.channels["downsample"]

    def test_zero_ratio_is_a_true_no_op(self, model, example):
        """ratio=0 must never trigger the alignment-relaxation machinery —
        there is nothing to relax around when nothing was requested."""
        _, stats = prune(model, example, ratio=0.0, align=16)
        assert stats.warnings == []
        assert stats.align_effective == stats.align
        assert all(before == after for before, after in stats.channels.values())

    def test_conflicting_group_overrides_resolve_to_the_gentlest_ratio(self):
        """`conv2` and `downsample` are tied into one Torch-Pruning group (see
        TinyResidualNet). Giving them different ratio_overrides must not let
        Torch-Pruning silently pick one at random — it should settle on the
        gentlest (most protective) of the two, applied to the whole group,
        and say so."""
        net = TinyResidualNet().eval()
        example = torch.randn(1, 3, 32, 32)
        overrides = {"conv2": 0.75, "downsample": 0.125}
        _, stats = prune(net, example, ratio=0.5, align=1, ratio_overrides=overrides)

        conv2_ratio = 1 - stats.channels["conv2"][1] / stats.channels["conv2"][0]
        downsample_ratio = 1 - stats.channels["downsample"][1] / stats.channels["downsample"][0]
        assert conv2_ratio == downsample_ratio == pytest.approx(0.125)
        assert any("conflict" in w and "gentlest" in w for w in stats.warnings)

    def test_single_channel_layer_warns_instead_of_crashing(self):
        """A 1-channel layer cannot be pruned at all — there is no alignment,
        however small, that removes a channel and leaves at least one behind.
        That must surface as a clear warning, not a crash and not a silent
        no-op indistinguishable from the alignment bug."""
        net = nn.Sequential(nn.Conv2d(3, 1, 3, padding=1), nn.ReLU(), nn.Conv2d(1, 10, 1))
        example = torch.randn(1, 3, 32, 32)
        pruned, stats = prune(net, example, ratio=0.9, align=16)

        assert stats.channels["0"] == (1, 1)
        assert any("cannot be pruned at all" in w for w in stats.warnings)
        assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10, 32, 32)

    def test_classifier_output_width_is_preserved(self, model, example):
        pruned, _ = prune(model, example, ratio=0.5, align=8)
        assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)
        assert pruned.classifier.out_features == 10

    def test_leaves_the_original_model_untouched(self, model, example):
        """The original is the distillation teacher — mutating it corrupts the run."""
        before = model.features[0].out_channels
        prune(model, example, ratio=0.5, align=8)
        assert model.features[0].out_channels == before

    def test_actually_removes_parameters(self, model, example):
        _, stats = prune(model, example, ratio=0.5, align=8)
        assert stats.params_after < stats.params_before
        assert 0.0 < stats.param_drop < 1.0

    def test_rejects_invalid_config(self):
        with pytest.raises(ValueError, match="ratio"):
            PruneConfig(ratio=1.5)
        with pytest.raises(ValueError, match="criterion"):
            PruneConfig(criterion="vibes")

    def test_classifier_detection(self, model):
        assert classifier_layers(model) == [model.classifier]

    def test_fully_convolutional_head_is_still_protected(self):
        """With no Linear anywhere, returning [] would leave the output layer
        prunable — which changes how many classes the model can predict."""
        net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 10, 1))
        assert classifier_layers(net) == [net[2]]


class TestPruneOnRealArchitectures:
    """The synthetic models above are built to hit one specific edge case
    each. The original bug report — whole ResNet stages silently left at 0%
    pruning — only ever showed up on the project's actual architectures,
    because the shared TinyCNN fixture (32/64-wide, no residual connections)
    never has a narrow, residual-tied layer to trigger it. This locks that
    down directly against every architecture models.build() can produce."""

    @pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
    @pytest.mark.parametrize("ratio", [0.3, 0.5, 0.7])
    def test_no_layer_is_silently_left_unpruned(self, arch, ratio):
        model = build(arch)
        example = torch.randn(1, 3, 32, 32)
        pruned, stats = prune(model, example, ratio=ratio, align=16)

        untouched = [
            name
            for name, (before, after) in stats.channels.items()
            if after == before and name not in ("fc", "classifier")
        ]
        assert not untouched, f"{arch} ratio={ratio}: {untouched} got 0% pruning silently"
        assert pruned(torch.randn(2, 3, 32, 32)).shape == (2, 10)


class TestRecover:
    def test_recomputes_statistics_and_restores_momentum(self, model, loader):
        bn = model.features[1]
        bn.running_mean.fill_(99.0)
        original_momentum = bn.momentum

        recalibrate_bn(model, loader)

        assert bn.running_mean.abs().max() < 10.0, "stale statistics were not replaced"
        assert bn.momentum == original_momentum, "momentum must be restored after calibration"

    def test_leaves_model_in_eval_mode(self, model, loader):
        model.eval()
        recalibrate_bn(model, loader)
        assert not model.training


class TestSensitivity:
    def test_ranks_every_quantizable_layer(self, model, loader):
        report = analyze(model, loader, batches=2)
        names = {layer.name for layer in report.layers}
        assert names == {"features.0", "features.3", "classifier"}

    def test_ranked_worst_first(self, model, loader):
        ranked = analyze(model, loader, batches=2).ranked()
        assert [layer.sqnr_db for layer in ranked] == sorted(layer.sqnr_db for layer in ranked)

    def test_prune_ratios_never_target_the_classifier(self, model, loader):
        """The classifier is in ignored_layers, so an override there goes nowhere."""
        report = analyze(model, loader, batches=2)
        assert "classifier" not in report.prune_ratios(0.5, protect=2)

    def test_prune_ratios_are_gentler_than_the_base(self, model, loader):
        report = analyze(model, loader, batches=2)
        assert all(r < 0.5 for r in report.prune_ratios(0.5).values())

    def test_exclude_nodes_are_real_onnx_names(self, model, loader, tmp_path):
        path = to_onnx(model, tmp_path / "m.onnx")
        import onnx

        graph_names = {n.name for n in onnx.load(str(path)).graph.node}
        excluded = analyze(model, loader, batches=2).exclude_nodes(path, k=2)
        assert excluded and set(excluded) <= graph_names

    def test_requires_calibration_data(self, model):
        with pytest.raises(ValueError, match="at least one"):
            analyze(model, [], batches=2)

    def test_exclude_nodes_raises_rather_than_silently_protecting_nothing(self, model, loader, tmp_path):
        """An empty exclusion list means 'exclude nothing' to the quantizer, so a
        lookup miss would drop the protection without saying so."""
        path = to_onnx(model, tmp_path / "m.onnx")
        report = analyze(model, loader, batches=2)
        report.layers[0].name = "layer.that.does.not.exist"

        with pytest.raises(KeyError, match="no matching node"):
            report.exclude_nodes(path, k=len(report.layers))


class TestExportAndMeasure:
    def test_export_supports_dynamic_batch(self, model, tmp_path):
        """One artifact must serve both batch-1 latency and batched accuracy."""
        from compresskit.core.benchmark import make_session
        import numpy as np

        path = to_onnx(model, tmp_path / "m.onnx")
        session = make_session(path)
        for batch in (1, 8):
            out = session.run(None, {"input": np.random.randn(batch, 3, 32, 32).astype(np.float32)})[0]
            assert out.shape == (batch, 10)

    def test_export_does_not_mutate_or_move_the_model(self, model, tmp_path):
        model.train()
        to_onnx(model, tmp_path / "m.onnx")
        assert model.training, "export must not leave the caller's model in eval mode"

    def test_benchmark_reports_ordered_percentiles(self, model, tmp_path):
        path = to_onnx(model, tmp_path / "m.onnx")
        latency = benchmark(path, warmup=3, runs=15)
        assert latency.median_ms > 0
        assert latency.p95_ms >= latency.median_ms
        assert latency.runs == 15 and latency.threads == 1
