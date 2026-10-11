# Budget-Search Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `compresskit.compress()` cheaper and more readable on real models: deduplicate repeated pruning warnings, skip prune trials that resolve to an already-measured architecture, keep the most sensitive layers in float32 as an automated step, and verify the QAT fallback end to end on a real checkpoint.

**Architecture:** All search changes live in `compresskit/search.py`. Mixed precision needs one new Pipeline option (`quantize(exclude_layers=[...])`), which maps torch layer names to ONNX node names *after* the pipeline exports its own graph. That mapping is factored into `compresskit/core/export.py` so `SensitivityReport.exclude_nodes` and the pipeline share one implementation.

**Tech Stack:** Python 3.14, PyTorch, Torch-Pruning, ONNX Runtime, pytest. Run everything with `./.venv/bin/python`.

**Spec:** No separate spec. The binding requirements are the user's four requests (recorded in the Goal above) plus the "honesty rules" section of `README.md`.

## Global Constraints

- Test command: `./.venv/bin/python -m pytest tests/ -q`. The baseline is 141 passing; every task ends with the full suite green.
- No new dependencies.
- Honesty rules from README.md stay enforced: latency is measured, never estimated; calibration data comes from the train split; accuracy is measured on the exported artifact; ops that stayed float32 are reported.
- `Attempt.prune_warnings` must keep returning the **full**, undeduplicated warning list. Deduplication is a display concern only.
- Match the existing comment style: comments explain *why*, in full sentences, the way the existing code in `search.py` and `pipeline.py` does.
- Tests use the synthetic fixtures in `tests/conftest.py` (`model`, `loader`, `example`). The test suite must never download CIFAR-10.
- Commit messages: imperative sentence-case subject (e.g. "Add QAT fallback and latency-lookup-table-guided pruning"), ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never stage `archive/System_Design.md`. It's the user's own uncommitted work.

## Review Focus

1. **The same warning, different numbers.** A ratio-conflict warning at ratio 0.225 and at 0.1125 is one issue. It should print once, not once per trial, and the suppression must be visible (a count), never silent. Owned by Task 1, which tests it.
2. **A duplicate-architecture skip must not change which attempt is chosen.** The reused verdict has to drive the binary search exactly as a fresh measurement would. Owned by Task 2, which tests that the search still escalates and still returns a best effort.
3. **Unknown layer names in `exclude_layers`.** These must raise `KeyError` naming the layer, never silently exclude nothing. Owned by Task 3, which tests it.
4. **`keep_float=0`.** This must disable the mixed-precision attempt entirely, with no sensitivity analysis spent on it. Owned by Task 3, which tests it.
5. **The QAT attempt must actually contain a QAT stage**, not just carry the label. Owned by Task 4, which tests it.

---

### Task 1: Deduplicate repeated pruning warnings in search output

**Files:**
- Modify: `compresskit/search.py` (the `log_prune_warnings` helper inside `compress()`, `SearchResult.summary()`, plus a new module-level helper)
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `compresskit.search._warning_key(warning: str) -> str`, and `compresskit.search._new_warnings(warnings: list[str], seen: set[str]) -> list[str]`, which returns the warnings whose key is not yet in `seen` and adds those keys to `seen`.

Context: during a budget search, every prune trial re-emits the same 3–4 warnings, differing only in the ratio number (e.g. "overridden to 0.225" vs "overridden to 0.1125"). On ResNet-20 that's ~20 long lines of repeats. A warning's *key* is its text with standalone decimal numbers masked. The live log and `SearchResult.summary()` both print a warning only the first time its key appears. `summary()` then ends with a count of the hidden repeats.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_search.py`, inside `class TestPruneWarnings`; also add `from compresskit.search import _new_warnings, _warning_key` at the top)

```python
    def test_warning_key_masks_ratio_numbers_but_not_layer_names(self):
        a = "['layer1.2.bn1'] share a connection with a layer overridden to 0.225 — using 0.225"
        b = "['layer1.2.bn1'] share a connection with a layer overridden to 0.1125 — using 0.1125"
        c = "['layer3.1.bn1'] share a connection with a layer overridden to 0.225 — using 0.225"
        assert _warning_key(a) == _warning_key(b), "same issue at a different ratio is one warning"
        assert _warning_key(a) != _warning_key(c), "a different layer is a different warning"

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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_search.py::TestPruneWarnings -v`
Expected: FAIL with `ImportError: cannot import name '_new_warnings'`

- [ ] **Step 3: Implement**

In `compresskit/search.py`, below `_merge_gentlest`:

```python
# A standalone decimal (a ratio, a millisecond figure) — but not the digits
# inside a layer path like "layer1.2.conv1", which identify *which* layer.
_STANDALONE_NUMBER = re.compile(r"(?<![\w.])\d+\.\d+")


def _warning_key(warning: str) -> str:
    """What makes two warnings the same issue: identical text once ratio
    numbers are masked. Every prune trial in a search re-emits the same
    conflict at a different ratio, and printing each copy buries the one
    line that is actually new."""
    return _STANDALONE_NUMBER.sub("#", warning)


def _new_warnings(warnings: list[str], seen: set[str]) -> list[str]:
    """The warnings whose key hasn't been shown yet; records them as shown."""
    fresh = []
    for warning in warnings:
        key = _warning_key(warning)
        if key not in seen:
            seen.add(key)
            fresh.append(warning)
    return fresh
```

In `SearchResult.summary()`, replace the per-attempt warning loop and add the count before the final verdict:

```python
    def summary(self) -> str:
        lines = [f"budget: {self.budget.describe()}", ""]
        seen: set[str] = set()
        hidden = 0
        for attempt in self.attempts:
            ...  # unchanged label / summary / efficiency / failures lines
            fresh = _new_warnings(attempt.prune_warnings, seen)
            hidden += len(attempt.prune_warnings) - len(fresh)
            for warning in fresh:
                lines.append(f"        ! {warning}")
        lines.append("")
        if hidden:
            lines.append(
                f"({hidden} repeated pruning warning(s) hidden — the same issue at a different "
                "ratio; every attempt's full list is in Attempt.prune_warnings)"
            )
        ...  # unchanged CHOSEN / BEST EFFORT lines
```

In `compress()`, create `shown_warnings: set[str] = set()` next to `attempts`, and change `log_prune_warnings` to print only `_new_warnings(candidate.prune_warnings, shown_warnings)`. Update its comment to say repeats are suppressed because the summary counts them.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_search.py -v`
Expected: all PASS (including the existing `test_alignment_warnings_reach_the_search_summary`)

- [ ] **Step 5: Run the full suite, then commit**

```bash
./.venv/bin/python -m pytest tests/ -q
git add compresskit/search.py tests/test_search.py
git commit -m "Print each repeated pruning warning once per search"
```

---

### Task 2: Skip prune trials that resolve to an already-measured architecture

**Files:**
- Modify: `compresskit/search.py` (the binary-search loop in `compress()`, plus a new module-level helper)
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `StructuredPruner`, `PruneConfig` from `compresskit.pruning` (already exported. Check `compresskit/pruning/__init__.py`). `PruneStats.channels` is `dict[str, tuple[int, int]]` (name → (before, after)). `PipelineResult.prune_stats` is that stats object's `__dict__`, so `attempt.result.prune_stats["channels"]` holds the same mapping.
- Produces: `compresskit.search._pruned_architecture(model, input_shape, ratio, align, ratio_overrides) -> tuple[tuple[str, int], ...]`, the sorted `(layer name, kept width)` pairs.

Context: SIMD alignment rounds each layer's kept-channel count to a multiple of 8/16, so neighbouring ratios often produce *identical* networks. In a real ResNet-20 run, ratios 0.338 and 0.394 both gave 168,450 parameters, and the second trial spent a full export + quantize + benchmark measuring the same network again. Pruning alone is cheap (no export, no ONNX, no benchmark). So for each ratio, compute the pruned architecture first. If an earlier trial already produced that architecture, reuse that trial's verdict for the binary search and don't record a new attempt.

- [ ] **Step 1: Write the failing test** (new class in `tests/test_search.py`)

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `./.venv/bin/python -m pytest tests/test_search.py::TestDuplicateArchitectures -v`
Expected: FAIL on the uniqueness assertion. With `TinyCNN` (widths 32 and 64, both convs protected at half ratio by the sensitivity overrides), ratios 0.787 and 0.844 should both keep 16 and 32 channels. **If it instead passes, or fails on `len(pruned) < 4` only, the fixture isn't producing a duplicate.** In that case, print each attempt's architecture and pick a `ratio_steps` value (or a small inline model, like the `narrow` model in `TestPruneWarnings`) that does, so the test is red before the fix. Record what you chose in the report.

- [ ] **Step 3: Implement**

Module-level helper in `compresskit/search.py` (add `StructuredPruner, PruneConfig` to the existing `from .pruning import ...` line):

```python
def _pruned_architecture(
    model: nn.Module, input_shape: tuple[int, ...], ratio: float, align: int,
    ratio_overrides: dict[str, float],
) -> tuple[tuple[str, int], ...]:
    """The network a prune trial would produce, without exporting or timing it.

    SIMD alignment rounds every layer's kept width to a block multiple, so
    nearby ratios routinely land on the identical network — and the expensive
    part of a trial is the export, quantize and benchmark, not the pruning.
    Same pruner, same config and same deterministic magnitude ranking as the
    Pipeline's prune stage, so the result is exactly what that trial builds.
    """
    pruner = StructuredPruner(
        model, torch.randn(*input_shape),
        PruneConfig(ratio=ratio, align=align, ratio_overrides=ratio_overrides),
    )
    pruner.run()
    return tuple(sorted((name, after) for name, (_, after) in pruner.stats().channels.items()))
```

In the binary-search loop, create `measured: dict[tuple, Attempt] = {}` before the loop. At the top of each iteration, after logging the `[2.N]` header:

```python
            architecture = _pruned_architecture(model, input_shape, ratio, align, merged_ratios(ratio))
            previous = measured.get(architecture)
            if previous is not None:
                # Identical network, so identical latency verdict — reuse it to
                # steer the search instead of paying for a second benchmark.
                log(f"    same architecture as {previous.label} — reusing its measurement")
                candidate = previous
            else:
                candidate = _attempt(...)  # unchanged
                attempts.append(candidate)
                measured[architecture] = candidate
                log(f"    {candidate.result.final.summary()}")
                log_prune_warnings(candidate)
                if hardest_pruned is None or ratio > hardest_pruned.config["ratio"]:
                    hardest_pruned = candidate
```

The rest of the iteration (`hits_speed`, `best_pruned`, `high`/`low` updates, the met → `finalize` early return) runs unchanged on `candidate`. A reused candidate that is `met` would already have returned when first measured, so the early return can't fire twice.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_search.py -v`
Expected: all PASS

- [ ] **Step 5: Run the full suite, then commit**

```bash
./.venv/bin/python -m pytest tests/ -q
git add compresskit/search.py tests/test_search.py
git commit -m "Skip prune trials that resolve to an already-measured architecture"
```

---

### Task 3: Keep the most sensitive layers in float32 as an automated search step

**Files:**
- Modify: `compresskit/core/export.py` (new `nodes_for_layers`)
- Modify: `compresskit/sensitivity/analysis.py` (`SensitivityReport.exclude_nodes` delegates to it)
- Modify: `compresskit/pipeline.py` (quantize stage accepts `exclude_layers`)
- Modify: `compresskit/search.py` (new mixed-precision attempt, new `keep_float` parameter, module docstring)
- Modify: `README.md` (the "Why this exists" sensitivity bullet and the Tier 1 paragraph)
- Test: `tests/test_pipeline.py`, `tests/test_search.py`

**Interfaces:**
- Produces: `compresskit.core.export.nodes_for_layers(onnx_path, layers: list[str]) -> list[str]`, which raises `KeyError` naming any layer with no node.
- Produces: `Pipeline.quantize(exclude_layers=[...torch layer names...])`.
- Produces: `compress(..., keep_float: int = 2)`. `0` disables the step. New attempt label: `"mixed-precision"`, with `config={"keep_float": [layer names]}`.

Context: the search can't pass `exclude_nodes` itself, because ONNX node names exist only after the pipeline exports its own graph. So the pipeline gets a layer-name option and maps it after export. In the search, the new step goes between quantize-only and QAT, and only runs when accuracy is the *only* failure. It costs one sensitivity analysis (forward passes only) and one quantize, with no training, so it is cheaper than QAT and is tried first.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pipeline.py` (match that file's existing import of `Pipeline`):

```python
class TestExcludeLayers:
    def test_named_layer_stays_float32(self, model, loader, tmp_path):
        result = (Pipeline(model, loader, loader, workdir=tmp_path, runs=4, warmup=1)
                  .quantize(precision="int8", exclude_layers=["features.0"])
                  .run(verbose=False))
        assert result.quantization.unquantized_ops.get("Conv", 0) >= 1

    def test_unknown_layer_is_an_error_not_a_silent_no_op(self, model, loader, tmp_path):
        pipe = Pipeline(model, loader, loader, workdir=tmp_path, runs=4, warmup=1)
        with pytest.raises(KeyError, match="no_such_layer"):
            pipe.quantize(precision="int8", exclude_layers=["no_such_layer"]).run(verbose=False)
```

If the host probe reports INT8 as not fast, `precision="int8"` raises. Pass `allow_slow_int8=True` in both calls so the tests are host-independent.

Append to `tests/test_search.py`, inside `class TestEscalation`:

```python
    def test_tries_mixed_precision_before_qat_when_only_accuracy_is_short(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", **FAST)
        labels = [a.label for a in result.attempts]
        assert labels[:2] == ["quantize-only", "mixed-precision"]
        assert labels.index("mixed-precision") < labels.index("qat+quantize")
        mixed = result.attempts[1]
        assert len(mixed.config["keep_float"]) == 2
        assert mixed.result.quantization.unquantized_ops, "the kept layers must actually stay float32"
        assert result.sensitivity is not None

    def test_keep_float_zero_skips_mixed_precision(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", keep_float=0, **FAST)
        assert "mixed-precision" not in [a.label for a in result.attempts]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_pipeline.py::TestExcludeLayers tests/test_search.py::TestEscalation -v`
Expected: FAIL. `exclude_layers` is an unexpected `QuantConfig` keyword, and `keep_float` is an unexpected `compress` keyword.

- [ ] **Step 3: Implement**

`compresskit/core/export.py`, below `layer_node_map`:

```python
def nodes_for_layers(path: str | Path, layers: list[str]) -> list[str]:
    """ONNX node names for the given torch layer paths.

    Raises rather than returning a short list: the quantizer treats a missing
    exclusion as "quantize it anyway", so a typo would silently remove the
    protection the caller asked for.
    """
    mapping = layer_node_map(path)
    nodes: list[str] = []
    for name in layers:
        found = mapping.get(name)
        if not found:
            raise KeyError(
                f"layer {name!r} has no matching node in {path}; known layers: {sorted(mapping)[:8]}"
            )
        nodes.extend(found)
    return nodes
```

`SensitivityReport.exclude_nodes` becomes `return nodes_for_layers(onnx_path, self.most_sensitive(k))`. Import `nodes_for_layers` in place of `layer_node_map` there if that's its only use. The existing comment about `[]` meaning "exclude nothing" moves into `nodes_for_layers`, already covered by its docstring above. Delete it from `exclude_nodes`.

`compresskit/pipeline.py`, in the quantize branch of `run()`, after `source = to_onnx(...)`:

```python
                options = dict(stage.options)
                # Layer names are what callers (and the sensitivity ranking)
                # know; node names only exist once this exact graph is
                # exported, so the mapping has to happen here, per run.
                exclude_layers = options.pop("exclude_layers", None)
                if exclude_layers:
                    options["exclude_nodes"] = [
                        *options.get("exclude_nodes", []),
                        *nodes_for_layers(source, exclude_layers),
                    ]
```

Then pass `QuantConfig(**options)` instead of `QuantConfig(**stage.options)`. Import `nodes_for_layers` from `.core.export`.

`compresskit/search.py`:
- Add `keep_float: int = 2` to `compress()`'s signature (after `qat_epochs`). Mention it in the docstring: how many of the most sensitive layers the mixed-precision step keeps in float32, with `0` to disable.
- Initialise `report: SensitivityReport | None = None` right after the quantize-only attempt. Compute the accuracy-only condition once, as `accuracy_only = budget.accuracy_failed(...) and not budget.latency_failed(...)` on that attempt. Reuse it for the existing QAT `if` (which currently also requires `train_loader is not None`).
- New step, after quantize-only fails and before the QAT block:

```python
    # 1.2. Accuracy is the only gap. Keeping the few layers the sensitivity
    #      ranking blames in float32 costs no training at all — cheaper than
    #      QAT, so it is tried first. It may give back some latency, which the
    #      measurement will show.
    if accuracy_only and keep_float > 0:
        log("\n[1.2] sensitivity analysis")
        report = analyze(model, calibration_loader, batches=4)
        log("    " + "\n    ".join(report.summary(top=5).splitlines()))
        kept = report.most_sensitive(keep_float)
        log(f"\n[1.3] mixed precision — keeping {kept} in float32")
        mixed = _attempt(
            "mixed-precision",
            f"quantize (keep {len(kept)} most sensitive in fp32)",
            lambda: run(pipeline("mixed").quantize(exclude_layers=kept)),
            budget,
            config={"keep_float": kept},
        )
        attempts.append(mixed)
        log(f"    {mixed.result.final.summary()}")
        if mixed.met:
            log("    budget met — stopping.")
            return finalize(mixed, report)
        for failure in mixed.failures:
            log(f"    - {failure}")
```

- The QAT block's `finalize(qat_attempt, None)` becomes `finalize(qat_attempt, report)`.
- Step 2 only analyzes if nothing has yet: wrap the existing `log("\n[2] sensitivity analysis")`, `report = analyze(...)` and summary log in `if report is None:`.
- Module docstring: insert a line between steps 1 and 2 describing mixed precision ("keep the most sensitive layers in float32, no training") and QAT as the accuracy-only fallbacks.

`README.md`:
- In the "Why this exists" sensitivity bullet, replace the text from "the same ranking can also name layers to keep in float32…" through "…not yet wired into the automated search itself." with: the same ranking also picks which layers the search keeps in float32 when quantization alone misses only on accuracy (`keep_float=`), and you can do the same by hand with `report.exclude_nodes(...)` (see Tier 3).
- In the Tier 1 paragraph, change "If only accuracy is short, it tries a brief QAT …" to say that if only accuracy is short, it first keeps the most sensitive layers in float32 (no training), then tries a brief QAT recovery pass.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_pipeline.py tests/test_search.py tests/test_core.py -v`
Expected: all PASS, including the existing `test_tries_qat_before_pruning_when_only_accuracy_is_short` and any existing `exclude_nodes` tests in `test_core.py`.

- [ ] **Step 5: Run the full suite, then commit**

```bash
./.venv/bin/python -m pytest tests/ -q
git add compresskit/core/export.py compresskit/sensitivity/analysis.py compresskit/pipeline.py compresskit/search.py README.md tests/test_pipeline.py tests/test_search.py
git commit -m "Keep the most sensitive layers in float32 as an automated search step"
```

---

### Task 4: Verify the QAT fallback end to end

**Files:**
- Test: `tests/test_search.py`
- No production code changes, unless the real run exposes a bug. In that case write a failing test that reproduces it first, then fix it.

**Interfaces:**
- Consumes: the `"mixed-precision"` and `"qat+quantize"` attempt labels from Task 3. Stage measurements are named `f"{index + 1}_{kind}"` (see `Pipeline.run`), so a QAT pipeline's stages include one whose `.stage` ends with `"_qat"`.

- [ ] **Step 1: Write the test** (inside `class TestEscalation`)

```python
    def test_qat_attempt_actually_runs_a_qat_stage(self, model, loader, tmp_path):
        """The label alone proves nothing — the measured stages must include QAT."""
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s", **FAST)
        qat_attempt = next(a for a in result.attempts if a.label == "qat+quantize")
        stage_names = [m.stage for m in qat_attempt.result.stages]
        assert any(name.endswith("_qat") for name in stage_names), stage_names
        assert stage_names[-1].endswith("_quantize"), "quantize must still be last"
```

- [ ] **Step 2: Run it**

Run: `./.venv/bin/python -m pytest tests/test_search.py::TestEscalation::test_qat_attempt_actually_runs_a_qat_stage -v`
Expected: PASS. This is a characterization test pinning existing behaviour. Prove it can fail by temporarily changing `.qat(epochs=qat_epochs)` to nothing in `search.py`'s QAT lambda, confirming FAIL, then reverting. Record both runs in the report.

- [ ] **Step 3: Real-checkpoint run**

Write a throwaway script **outside the repo** (in your report directory) that loads the real ResNet-20 checkpoint and runs `compresskit.compress` with a budget where only accuracy fails. Run it with the branch's code first on the path: `PYTHONPATH="$PWD" ./.venv/bin/python <script>`.
- Use the existing loaders and checkpoint I/O: `compresskit/zoo/data.py` (`calibration_loader`, `test_loader`, `train_loader`) and `compresskit/zoo/train.py` (checkpoint loading). Read those files for exact signatures. The checkpoint is `checkpoints/resnet20_fp32.pt`, and CIFAR-10 is already in `data/`.
- Pick a budget like `latency<5ms, accuracy_drop<X%`, where X is below the measured quantize-only drop (quantize-only measured 91.54% INT8 earlier today). Read the fp32 baseline from the quantize-only attempt's `result.baseline.accuracy`. If the INT8 drop is too small for any meaningful X (under ~0.1 pp), use an absolute `accuracy>` floor just above the INT8 number instead.
- Use `qat_epochs=1` to bound runtime, and `run_in_background` if it exceeds a few minutes.
- Required outcome: the log shows `[1.3] mixed precision` and `[1.5] QAT recovery` both running without exceptions. Whether the budget is *met* doesn't matter. Put in the report: the budget used, each attempt's accuracy / latency / size, and total runtime.

- [ ] **Step 4: Run the full suite, then commit**

```bash
./.venv/bin/python -m pytest tests/ -q
git add tests/test_search.py
git commit -m "Pin that the QAT attempt really runs a QAT stage"
```

---

### Task 5: Re-measure borderline latency passes before trusting them

**Files:**
- Modify: `compresskit/search.py` (`Attempt`, `_remeasure_full`, `compress()`)
- Test: `tests/test_search.py`

**Interfaces:**
- Produces: `Attempt.full_fidelity: bool = False`, which `_remeasure_full` sets to True. `_remeasure_full` becomes a no-op on an attempt that is already full-fidelity.
- Produces: `compress(..., confirm_margin: float = 0.1)`.

Context: search measurements use `search_runs` (default 60) timed runs, so a candidate near the limit can pass on noise. In a real ResNet-20 run, prune@0.450 measured 0.249 ms against a 0.25 ms budget (p95 0.369 ms). The search accepted it as fast enough, and the same network later re-measured at 0.296–0.397 ms. Fix: when a candidate *passes* the latency budget with a median inside the top `confirm_margin` fraction of the budget (`median > (1 - confirm_margin) * max_latency_ms`), re-measure it at full fidelity before any decision uses the verdict. A clear pass or any fail is never re-measured; failing is never the risky direction here. With no latency constraint in the budget, nothing is borderline.

- [ ] **Step 1: Write the failing tests** (new class in `tests/test_search.py`)

```python
class TestBorderlineLatency:
    def test_borderline_pass_is_remeasured_before_it_steers_the_search(self, model, loader, tmp_path):
        # confirm_margin≈1 makes any latency pass borderline, so the
        # quantize-only attempt (passes latency, fails accuracy) must be
        # confirmed at full fidelity even though the search moves past it.
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s",
                          confirm_margin=0.999999, **FAST)
        first = result.attempts[0]
        assert first.label == "quantize-only"
        assert first.full_fidelity
        assert first.result.final.latency.runs == 200

    def test_clear_passes_are_not_remeasured(self, model, loader, tmp_path):
        result = compress(model, loader, loader, budget="latency<1000ms,accuracy>99.99%",
                          train_loader=loader, workdir=tmp_path / "s",
                          confirm_margin=0.0, **FAST)
        for attempt in result.attempts:
            if attempt is not result.chosen:
                assert not attempt.full_fidelity, attempt.label
                assert attempt.result.final.latency.runs == FAST["search_runs"]

    def test_chosen_attempt_is_not_measured_twice(self, model, loader, tmp_path, monkeypatch):
        import compresskit.search as search
        calls = []
        original = search.measure
        monkeypatch.setattr(search, "measure", lambda *a, **k: calls.append(1) or original(*a, **k))
        compress(model, loader, loader, budget="latency<1000ms",
                 workdir=tmp_path / "s", confirm_margin=0.999999, **FAST)
        # quantize-only meets this budget: confirmed once (baseline + 1 stage),
        # and finalize must not repeat that work.
        assert len(calls) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./.venv/bin/python -m pytest tests/test_search.py::TestBorderlineLatency -v`
Expected: FAIL with `TypeError: compress() got an unexpected keyword argument 'confirm_margin'`

- [ ] **Step 3: Implement**

- `Attempt` gets `full_fidelity: bool = False`.
- `_remeasure_full`: return immediately if `attempt.full_fidelity`. Set it to True at the end.
- `compress()`: add `confirm_margin: float = 0.1` after `search_accuracy_limit`, and document it. Add a local helper next to `log_prune_warnings`:

```python
    def confirm_borderline(candidate: Attempt) -> Attempt:
        # A pass within the measurement noise of the limit is not a pass yet:
        # the search times few runs to stay affordable, and a lucky median at
        # the boundary would steer every later decision. Only passes are
        # checked — a borderline *fail* just costs one more trial, while a
        # false pass is how a budget-missing network gets chosen.
        limit = budget.max_latency_ms
        if limit is None or candidate.full_fidelity:
            return candidate
        median = candidate.result.final.latency.median_ms
        if median <= limit and median > (1.0 - confirm_margin) * limit:
            log(f"    borderline ({median:.3f} ms vs {limit:.3f} ms) — re-measuring at full fidelity")
            _remeasure_full(candidate, budget, test_loader, input_shape)
            log(f"    {candidate.result.final.summary()}")
        return candidate
```

- Wrap every `_attempt(...)` call in `compress()` (quantize-only, mixed-precision, qat, each prune trial, distilled) as `confirm_borderline(_attempt(...))`, so the confirmed verdict is what every following `met` / `latency_failed` check sees.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./.venv/bin/python -m pytest tests/test_search.py -v`
Expected: all PASS, including `test_chosen_result_is_remeasured_at_full_fidelity`

- [ ] **Step 5: Run the full suite, then commit**

```bash
./.venv/bin/python -m pytest tests/ -q
git add compresskit/search.py tests/test_search.py
git commit -m "Re-measure borderline latency passes before trusting them"
```
