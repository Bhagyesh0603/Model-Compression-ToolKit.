"""The escalation logic: cheapest stage that works, and stop there."""

import pytest
import torch.nn as nn

from compresskit.search import compress

FAST = dict(search_runs=4, search_accuracy_limit=64, ratio_steps=1, distill_epochs=1, verbose=False)


class TestEscalation:
    def test_stops_at_quantize_only_when_that_is_enough(self, model, loader, tmp_path):
        """The headline behaviour: never run pruning or fine-tuning you didn't need."""
        result = compress(model, loader, loader, budget="size<10mb",
                          workdir=tmp_path / "s", **FAST)

        assert result.met
        assert len(result.attempts) == 1
        assert result.chosen.label == "quantize-only"
        assert result.sensitivity is None, "sensitivity analysis is wasted work if we stopped early"

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
