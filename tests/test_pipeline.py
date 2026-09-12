"""Integration tests: the stages actually compose, and the guardrails hold."""

import pytest
import torch

from compresskit.core import host as host_mod
from compresskit.core.export import to_onnx
from compresskit.core.host import HostContext
from compresskit.pipeline import Pipeline
from compresskit.quantization import QuantConfig, quantize
from compresskit.report import render


@pytest.fixture
def pipeline(model, loader, tmp_path):
    return Pipeline(model, loader, loader, train_loader=loader, workdir=tmp_path / "artifacts",
                    warmup=2, runs=8)


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

        from compresskit.distill import DistillConfig, distill

        teacher = model
        student = copy.deepcopy(model)
        teacher.train()

        distill(student, teacher, loader, DistillConfig(epochs=1, device="cpu"))

        assert all(p.requires_grad for p in teacher.parameters())
        assert teacher.training

    def test_pipeline_does_not_mutate_the_input_model(self, model, loader, tmp_path):
        before = model.features[0].out_channels
        Pipeline(model, loader, loader, workdir=tmp_path / "a", warmup=1, runs=4) \
            .prune(0.5, align=8).run(verbose=False)
        assert model.features[0].out_channels == before
