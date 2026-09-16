"""Integration tests: the stages actually compose, and the guardrails hold."""

import pytest
import torch

from compresskit.core import host as host_mod
from compresskit.core.benchmark import Latency, Measurement
from compresskit.core.export import to_onnx
from compresskit.core.host import HostContext
from compresskit.pipeline import Pipeline, PipelineResult
from compresskit.quantization import QATConfig, QuantConfig, qat, quantize
from compresskit.report import render, sensitivity_figure
from compresskit.sensitivity.analysis import LayerSensitivity, SensitivityReport
from compresskit.zoo.models import build


@pytest.fixture
def pipeline(model, loader, tmp_path):
    return Pipeline(model, loader, loader, train_loader=loader, workdir=tmp_path / "artifacts",
                    warmup=2, runs=8)


def _latency(median_ms=1.0):
    return Latency(median_ms=median_ms, p95_ms=median_ms, mean_ms=median_ms, stdev_ms=0.0,
                    runs=1, warmup=0, threads=1, batch=1)


def _measurement(stage="stage", accuracy=0.9, median_ms=1.0, size_mb=1.0, precision="fp32"):
    return Measurement(stage=stage, precision=precision, accuracy=accuracy, latency=_latency(median_ms),
                       size_mb=size_mb, parameters=100, artifact="m.onnx", host={"cpu": "test"})


class TestOrdering:
    def test_quantize_must_be_last(self, pipeline):
        """Quantization changes weight dtype — pruning after it is broken, not slow."""
        pipeline.quantize().prune(0.5)
        with pytest.raises(ValueError, match="quantize must be the last stage"):
            pipeline.run(verbose=False)

    def test_preflight_warns_about_bad_ordering_before_spending_compute(self, pipeline):
        pipeline.quantize().prune(0.5)
        assert "quantize is not last" in pipeline.preflight()

    def test_preflight_reports_host_and_plan(self, pipeline):
        text = pipeline.prune(0.5).recover().quantize().preflight()
        assert "prune -> recover -> quantize" in text
        assert "classifier" in text
        assert "calibration" in text


class TestQuantize:
    def test_int8_shrinks_the_file_and_covers_the_compute_ops(self, model, loader, tmp_path):
        source = to_onnx(model, tmp_path / "fp32.onnx")
        result = quantize(source, tmp_path / "int8.onnx", loader, QuantConfig(precision="int8"))

        assert result.precision == "int8"
        assert result.size_mb < source.stat().st_size / (1024 * 1024)
        assert result.coverage > 0.5, "most compute ops should have an integer path"

    def test_fp16_is_flagged_as_a_size_only_win(self, model, loader, tmp_path):
        source = to_onnx(model, tmp_path / "fp32.onnx")
        result = quantize(source, tmp_path / "fp16.onnx", loader, QuantConfig(precision="fp16"))
        assert any("size win, not a speed win" in w for w in result.warnings)

    def test_excluded_nodes_are_reported_as_still_float(self, model, loader, tmp_path):
        """Coverage must count only ops fused into integer kernels. An excluded
        node still consumes the previous pair's DequantizeLinear, so an
        input-side-only check would count it as quantized and hide the fact that
        the file shrank without that op getting any faster."""
        from compresskit.core.export import layer_node_map

        source = to_onnx(model, tmp_path / "fp32.onnx")
        excluded = layer_node_map(source)["features.3"]

        result = quantize(source, tmp_path / "int8.onnx", loader,
                          QuantConfig(precision="int8", exclude_nodes=excluded))

        assert result.unquantized_ops, "the excluded conv should be reported as still float"
        assert result.coverage < 1.0
        assert any("did not get faster" in w for w in result.warnings)

    def test_int8_requires_calibration_data(self, model, tmp_path):
        source = to_onnx(model, tmp_path / "fp32.onnx")
        with pytest.raises(ValueError, match="calibration loader"):
            quantize(source, tmp_path / "int8.onnx", None, QuantConfig(precision="int8"))

    def test_int8_is_refused_on_a_cpu_without_fused_kernels(self, model, loader, tmp_path, monkeypatch):
        """Shipping INT8 to a pre-VNNI CPU is a latency regression; make it deliberate."""
        slow = HostContext(system="Linux", machine="x86_64", cpu="Old Xeon", logical_cores=4,
                           performance_cores=4, features=frozenset({"avx2"}), versions={})
        monkeypatch.setattr(host_mod, "probe", lambda: slow)

        with pytest.raises(RuntimeError, match="allow_slow_int8"):
            quantize(to_onnx(model, tmp_path / "fp32.onnx"), tmp_path / "int8.onnx", loader,
                     QuantConfig(precision="int8"))

    def test_auto_precision_follows_the_host(self, model, loader, tmp_path, monkeypatch):
        slow = HostContext(system="Linux", machine="x86_64", cpu="Old Xeon", logical_cores=4,
                           performance_cores=4, features=frozenset({"avx2"}), versions={})
        monkeypatch.setattr(host_mod, "probe", lambda: slow)

        result = quantize(to_onnx(model, tmp_path / "fp32.onnx"), tmp_path / "auto.onnx", loader,
                          QuantConfig(precision="auto"))
        assert result.precision == "fp16"

    def test_excluding_a_residual_add_fed_conv_actually_works(self, loader, tmp_path):
        """The shared `model` fixture (TinyCNN) has no residual connections,
        so it can never exercise this: ONNX Runtime's quantizer decides
        whether to quantize a Conv's output based on what its *consumer*
        needs, and a residual Add still wants quantized inputs — naming just
        the Conv in exclude_nodes used to be silently ignored."""
        net = build("resnet20")
        source = to_onnx(net, tmp_path / "fp32.onnx", (1, 3, 32, 32))
        excluded = "/layer3/layer3.2/conv2/Conv"

        result = quantize(source, tmp_path / "int8.onnx", loader,
                          QuantConfig(exclude_nodes=[excluded]))

        assert result.unquantized_ops == {"Conv": 1}, "the excluded conv should stay float32"
        assert result.coverage < 1.0
        assert any("residual Add" in w for w in result.warnings)

    def test_excluding_one_branch_does_not_collaterally_protect_its_sibling(self, loader, tmp_path):
        """A block-0 residual Add has two Conv inputs (the main path and its
        downsample shortcut). Excluding only the main path's conv must not
        leave the downsample conv unquantized too as a side effect of also
        excluding the shared Add."""
        net = build("resnet20")
        source = to_onnx(net, tmp_path / "fp32.onnx", (1, 3, 32, 32))

        result = quantize(source, tmp_path / "int8.onnx", loader,
                          QuantConfig(exclude_nodes=["/layer2/layer2.0/conv2/Conv"]))

        assert result.unquantized_ops == {"Conv": 1}, "only the requested conv should stay float32"


class TestFullRun:
    def test_every_stage_is_measured(self, pipeline, tmp_path):
        result = pipeline.prune(0.5, align=8).recover().quantize().run(verbose=False)

        assert [m.stage for m in result.stages] == ["1_prune", "2_recover", "3_quantize"]
        assert result.final.precision in ("int8", "fp16")
        assert result.final.parameters < result.baseline.parameters
        assert all(m.latency.median_ms > 0 for m in result.stages)
        # Every measurement carries the host it was taken on.
        assert result.baseline.host["cpu"] == result.host["cpu"]

    def test_distillation_needs_training_data(self, model, loader, tmp_path):
        pipe = Pipeline(model, loader, loader, workdir=tmp_path / "a", warmup=1, runs=4)
        with pytest.raises(ValueError, match="distillation needs training data"):
            pipe.distill(epochs=1).run(verbose=False)

    def test_distill_rejects_an_empty_loader_instead_of_training_on_nothing(self, model):
        """An empty loader used to let `distill` "complete" every epoch
        having iterated zero batches — no error, no warning, and the student
        came back byte-identical to how it went in."""
        from torch.utils.data import DataLoader, TensorDataset

        from compresskit.distillation import DistillConfig, distill

        empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
        with pytest.raises(ValueError, match="at least one training batch"):
            distill(model, model, empty_loader, DistillConfig(epochs=2, device="cpu"))

    def test_distill_stage_never_silently_substitutes_an_explicit_empty_loader(self, model, loader, tmp_path):
        """A loader with zero batches is falsy in Python, so `explicit or
        self.train_loader` used to quietly fall back to the pipeline's own
        train_loader instead of the (empty) one the caller actually passed
        to `.distill(loader=...)` — training on the wrong data with no sign
        anything was substituted."""
        from torch.utils.data import DataLoader, TensorDataset

        empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
        pipe = Pipeline(model, loader, loader, train_loader=loader, workdir=tmp_path / "a", warmup=1, runs=4)
        with pytest.raises(ValueError, match="at least one training batch"):
            pipe.distill(epochs=1, loader=empty_loader).run(verbose=False)

    def test_result_serialises_and_renders(self, pipeline, tmp_path):
        result = pipeline.prune(0.5, align=8).recover().quantize().run(verbose=False)

        payload = result.as_dict()
        assert payload["totals"]["speedup"] > 0
        assert payload["host"]["cpu"]

        json_path = result.save_json(tmp_path / "result.json")
        assert json_path.exists() and json_path.stat().st_size > 0

        html_path = render(result, tmp_path / "report.html")
        text = html_path.read_text()
        assert "data:image/png;base64," in text, "charts must be inlined for a standalone file"
        assert "Per-stage measurements" in text

    def test_distillation_hands_the_teacher_back_unchanged(self, model, loader):
        """`distill` is public API — freezing the caller's model permanently
        would make it silently untrainable afterwards."""
        import copy

        from compresskit.distillation import DistillConfig, distill

        teacher = model
        student = copy.deepcopy(model)
        teacher.train()

        distill(student, teacher, loader, DistillConfig(epochs=1, device="cpu"))

        assert all(p.requires_grad for p in teacher.parameters())
        assert teacher.training

    def test_teacher_is_restored_even_when_the_run_fails_partway_through(self, model, loader):
        """The happy-path restoration above only proves the normal-completion
        case works. Without a try/finally around the training loop, a
        mid-run exception (a flaky on_epoch callback, a bad batch, a
        KeyboardInterrupt) left the caller's teacher permanently frozen and
        in eval mode -- the exact outcome `distill`'s own docstring promises
        never happens."""
        import copy

        from compresskit.distillation import DistillConfig, distill

        teacher = model
        student = copy.deepcopy(model)
        teacher.train()

        def flaky_on_epoch(student, epoch, stats):
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="simulated failure"):
            distill(student, teacher, loader, DistillConfig(epochs=2, device="cpu"), on_epoch=flaky_on_epoch)

        assert all(p.requires_grad for p in teacher.parameters())
        assert teacher.training

    def test_pipeline_does_not_mutate_the_input_model(self, model, loader, tmp_path):
        before = model.features[0].out_channels
        Pipeline(model, loader, loader, workdir=tmp_path / "a", warmup=1, runs=4) \
            .prune(0.5, align=8).run(verbose=False)
        assert model.features[0].out_channels == before


class TestQAT:
    """QAT trains against simulated quantization noise to recover accuracy
    plain PTQ alone loses -- mirrors distill()'s test density since it
    follows the exact same structural contract (empty-loader guard, borrowed-
    state restoration on failure)."""

    def test_qat_rejects_an_empty_loader_instead_of_training_on_nothing(self, model):
        from torch.utils.data import DataLoader, TensorDataset

        empty_loader = DataLoader(TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)))
        with pytest.raises(ValueError, match="at least one training batch"):
            qat(model, empty_loader, QATConfig(epochs=2, device="cpu"))

    def test_forward_is_restored_even_when_training_fails_partway_through(self, model, loader):
        """Without restoration, a mid-run exception would leave every Conv2d/
        Linear's forward permanently patched -- a later `to_onnx` export would
        then silently trace the fake-quantized graph instead of the real one."""
        original_conv_forward = model.features[0].forward.__func__

        def flaky_on_epoch(m, epoch, stats):
            raise RuntimeError("simulated failure")

        with pytest.raises(RuntimeError, match="simulated failure"):
            qat(model, loader, QATConfig(epochs=2, device="cpu"), on_epoch=flaky_on_epoch)

        assert model.features[0].forward.__func__ is original_conv_forward

    def test_pipeline_qat_then_quantize_produces_a_measured_stage(self, model, loader, tmp_path):
        result = Pipeline(model, loader, loader, train_loader=loader, workdir=tmp_path / "a", warmup=1, runs=4) \
            .qat(epochs=1).quantize().run(verbose=False)
        assert [m.stage for m in result.stages] == ["1_qat", "2_quantize"]

    def test_qat_stage_needs_training_data(self, model, loader, tmp_path):
        pipe = Pipeline(model, loader, loader, workdir=tmp_path / "a", warmup=1, runs=4)
        with pytest.raises(ValueError, match="QAT needs training data"):
            pipe.qat(epochs=1).run(verbose=False)


class TestReportRendering:
    def test_sensitivity_chart_flags_the_worst_layer_when_sqnr_is_negative(self):
        """`worst * 1.05` as a "within 5% of worst" threshold flips direction
        once `worst` is negative (noise > signal — the case this chart exists
        to flag), so no bar, not even the actual worst one, used to render red."""
        report = SensitivityReport(
            layers=[
                LayerSensitivity(name="a", kind="conv", parameters=1, mse=0.0, sqnr_db=12.0),
                LayerSensitivity(name="b", kind="conv", parameters=1, mse=0.0, sqnr_db=-6.0),
                LayerSensitivity(name="c", kind="conv", parameters=1, mse=0.0, sqnr_db=-8.0),
            ],
            bits=8, batches=1,
        )
        figure = sensitivity_figure(report)
        axis = figure.axes[0]
        # ranked() sorts worst (lowest SQNR) first, then the chart reverses for
        # bottom-to-top display, so bar order is [a, b, c] on the y-axis.
        colours = [bar.get_facecolor() for bar in axis.patches]
        from compresskit.report import ROSE
        import matplotlib.colors as mcolors
        rose_rgb = mcolors.to_rgba(ROSE, alpha=0.8)
        assert colours[2] == pytest.approx(rose_rgb), "the actual worst layer (c, -8dB) must be flagged rose"
        assert colours[0] != pytest.approx(rose_rgb), "the best layer must not be flagged"

    def test_size_card_is_not_green_when_the_model_grew(self, tmp_path):
        baseline = _measurement(stage="baseline", size_mb=1.0)
        grown = _measurement(stage="1_stage", size_mb=1.5)
        result = PipelineResult(baseline=baseline, stages=[grown], host={"cpu": "test"})

        html = render(result, tmp_path / "report.html").read_text()
        size_card = html.split('<div class="label">size</div>')[1].split("</div></div>")[0]
        assert 'class="value good"' not in size_card
        assert 'class="value warn"' in size_card

    def test_speedup_and_size_reduction_do_not_divide_by_zero(self, tmp_path):
        baseline = _measurement(stage="baseline", median_ms=5.0, size_mb=2.0)
        zero_latency = _measurement(stage="1_stage", median_ms=0.0, size_mb=0.0)
        result = PipelineResult(baseline=baseline, stages=[zero_latency], host={"cpu": "test"})

        assert result.speedup > 0 and result.speedup < float("inf")
        assert result.size_reduction > 0 and result.size_reduction < float("inf")
        # Must not raise, and must produce a finite (if huge) rendered report.
        html = render(result, tmp_path / "report.html").read_text()
        assert "inf" not in html.lower().replace("info", "")
