"""Self-contained HTML report — the per-run artifact a human actually reads.

Charts are rendered to PNG and inlined as base64 so the report is a single file
you can email or attach to a PR with no asset directory to lose.

The Pareto view plots accuracy against *measured* latency, never FLOPs. Two
models with identical FLOPs routinely differ 2x in wall-clock, so a FLOPs-based
Pareto front recommends the wrong model with total confidence.
"""

from __future__ import annotations

import base64
import html
import io
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .core.benchmark import Measurement, efficiency_score  # noqa: E402

SLATE = "#334155"
MUTED = "#94A3B8"
VIOLET = "#6D28D9"
TEAL = "#0F766E"
ROSE = "#B91C1C"
AMBER = "#B45309"

_CSS = """
:root{--slate-50:#F8FAFC;--slate-100:#F1F5F9;--slate-200:#E2E8F0;--slate-400:#94A3B8;
--slate-500:#64748B;--slate-700:#334155;--slate-900:#0F172A;--violet:#6D28D9;
--teal:#0F766E;--rose:#B91C1C;--amber:#B45309;--green:#047857;}
*{box-sizing:border-box}
body{margin:0;padding:40px 28px 80px;background:#fff;color:var(--slate-700);
font:15.5px/1.7 'Segoe UI',-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:27px;color:var(--slate-900);margin:0 0 6px;letter-spacing:-.4px}
h2{font-size:16px;color:var(--slate-900);margin:44px 0 14px;padding-bottom:8px;
border-bottom:1px solid var(--slate-200);letter-spacing:-.2px}
.sub{color:var(--slate-500);font-size:13.5px;margin:0 0 4px}
.mono{font-family:'SF Mono',SFMono-Regular,ui-monospace,Menlo,Consolas,monospace}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:26px 0 0}
.card{border:1px solid var(--slate-200);border-radius:10px;padding:16px 18px;background:var(--slate-50)}
.card .label{font-size:10.5px;letter-spacing:1.2px;text-transform:uppercase;color:var(--slate-400);
font-weight:700;margin-bottom:7px}
.card .value{font-size:25px;font-weight:600;color:var(--slate-900);letter-spacing:-.5px;
font-variant-numeric:tabular-nums}
.card .note{font-size:12px;color:var(--slate-500);margin-top:3px}
.good{color:var(--green)!important}.bad{color:var(--rose)!important}.warn{color:var(--amber)!important}
table{width:100%;border-collapse:collapse;font-size:13.5px;margin-top:6px}
th{text-align:left;font-size:10.5px;letter-spacing:1px;text-transform:uppercase;
color:var(--slate-400);font-weight:700;padding:0 12px 9px;border-bottom:1px solid var(--slate-200)}
td{padding:10px 12px;border-bottom:1px solid var(--slate-100);
font-variant-numeric:tabular-nums}
tr.baseline td{background:var(--slate-50);font-weight:600}
tr:last-child td{border-bottom:none}
td.num{text-align:right}
img{max-width:100%;border:1px solid var(--slate-200);border-radius:10px;margin-top:8px}
.note-list{list-style:none;padding:0;margin:12px 0 0}
.note-list li{padding:10px 14px;border-left:3px solid var(--amber);background:#FFFBEB;
border-radius:0 6px 6px 0;margin-bottom:8px;font-size:13.5px}
.kv{display:grid;grid-template-columns:170px 1fr;gap:5px 16px;font-size:13px;margin-top:8px}
.kv dt{color:var(--slate-400);font-weight:600}
.kv dd{margin:0;color:var(--slate-700)}
footer{margin-top:56px;padding-top:18px;border-top:1px solid var(--slate-200);
font-size:12px;color:var(--slate-400)}
"""


def _png_data_uri(figure) -> str:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=144, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#CBD5E1")
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(True, alpha=0.25, linewidth=0.7, color="#CBD5E1")
    axis.set_axisbelow(True)


def pareto_figure(measurements: list[Measurement]):
    """Accuracy vs measured latency; marker area is disk size."""
    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    _style(axis)

    sizes = [max(m.size_mb, 0.01) for m in measurements]
    scale = 900 / max(sizes)
    for index, m in enumerate(measurements):
        colour = SLATE if index == 0 else (VIOLET if m.precision == "int8" else TEAL)
        axis.scatter(
            m.latency.median_ms, m.accuracy * 100,
            s=max(m.size_mb * scale, 60), alpha=0.55, color=colour, edgecolors="white", linewidth=1.6, zorder=3,
        )
        axis.annotate(
            m.stage, (m.latency.median_ms, m.accuracy * 100),
            textcoords="offset points", xytext=(0, 15), ha="center", fontsize=8.5, color=SLATE,
        )

    axis.set_xlabel("measured latency (ms, median, batch 1, 1 thread)", fontsize=10, color=SLATE)
    axis.set_ylabel("top-1 accuracy (%)", fontsize=10, color=SLATE)
    axis.set_title("accuracy vs measured latency — marker area is disk size",
                   fontsize=11, color=SLATE, pad=14, loc="left")
    return figure


def sensitivity_figure(report):
    """Per-layer SQNR, worst first. Low bars are the layers to protect."""
    ranked = report.ranked()
    figure, axis = plt.subplots(figsize=(7.6, max(2.6, 0.26 * len(ranked))))
    _style(axis)

    names = [layer.name for layer in ranked][::-1]
    values = [layer.sqnr_db for layer in ranked][::-1]
    worst = min(values) if values else 0
    # "Within 5% of the worst" has to shift the threshold *toward* worst
    # regardless of its sign: `worst * 1.05` moves a negative worst further
    # negative (inverting the comparison), so scale by magnitude instead.
    threshold = worst + abs(worst) * 0.05
    colours = [ROSE if v <= threshold else VIOLET for v in values]

    axis.barh(range(len(values)), values, color=colours, alpha=0.8, height=0.72)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=8.5, color=SLATE)
    axis.set_xlabel(f"SQNR (dB) under {report.bits}-bit weights — lower means more damage",
                    fontsize=10, color=SLATE)
    axis.set_title("per-layer quantization sensitivity", fontsize=11, color=SLATE, pad=14, loc="left")
    return figure


def _card(label: str, value: str, note: str = "", tone: str = "") -> str:
    klass = f" {tone}" if tone else ""
    return (
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value{klass}">{html.escape(value)}</div>'
        f'<div class="note">{html.escape(note)}</div></div>'
    )


def _stage_rows(baseline: Measurement, stages: list[Measurement]) -> str:
    rows = []
    for index, m in enumerate([baseline] + stages):
        drop = (baseline.accuracy - m.accuracy) * 100
        speed = baseline.latency.median_ms / max(m.latency.median_ms, 1e-9)
        rows.append(
            f'<tr class="{"baseline" if index == 0 else ""}">'
            f"<td class='mono'>{html.escape(m.stage)}</td>"
            f"<td>{html.escape(m.precision)}</td>"
            f'<td class="num">{m.accuracy * 100:.2f}%</td>'
            f'<td class="num">{"—" if index == 0 else f"{-drop:+.2f}"}</td>'
            f'<td class="num">{m.latency.median_ms:.3f}</td>'
            f'<td class="num">{m.latency.p95_ms:.3f}</td>'
            f'<td class="num">{"—" if index == 0 else f"{speed:.2f}x"}</td>'
            f'<td class="num">{m.size_mb:.2f}</td>'
            f'<td class="num">{m.parameters:,}</td></tr>'
        )
    return "\n".join(rows)


def render(result, path: str | Path, sensitivity=None, title: str = "Compression report") -> Path:
    """Write a standalone HTML report for a PipelineResult. Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    baseline, final = result.baseline, result.final
    host = result.host
    drop = result.accuracy_drop

    cards = "".join([
        _card("accuracy", f"{final.accuracy * 100:.2f}%",
              f"{-drop:+.2f} pp vs baseline", "bad" if drop > 1.0 else "good"),
        _card("latency", f"{final.latency.median_ms:.3f} ms",
              f"{result.speedup:.2f}x vs {baseline.latency.median_ms:.3f} ms",
              "good" if result.speedup > 1.05 else "warn"),
        _card("size", f"{final.size_mb:.2f} MB",
              f"{result.size_reduction:.2f}x vs {baseline.size_mb:.2f} MB",
              "good" if result.size_reduction > 1.0 else "warn"),
        _card("efficiency score", f"{result.efficiency_score:.4f}",
              "harmonic mean of accuracy and compression"),
    ])

    charts = f'<h2>Pareto — accuracy vs measured latency</h2><img src="{_png_data_uri(pareto_figure([baseline] + result.stages))}" alt="pareto">'
    if sensitivity is not None:
        charts += f'<h2>Per-layer sensitivity</h2><img src="{_png_data_uri(sensitivity_figure(sensitivity))}" alt="sensitivity">'

    warnings = ""
    if result.warnings:
        items = "".join(f"<li>{html.escape(w)}</li>" for w in result.warnings)
        warnings = f'<h2>Notes</h2><ul class="note-list">{items}</ul>'

    versions = ", ".join(f"{k} {v}" for k, v in host.get("versions", {}).items())
    quant = result.quantization
    quant_row = ""
    if quant:
        # Op coverage counts integer kernels, so it is only meaningful for INT8;
        # printing "0% of compute ops in low precision" for an FP16 model would
        # read as a failure rather than a category error.
        detail = (
            f", {quant.coverage * 100:.0f}% of compute ops in integer kernels"
            if quant.precision == "int8"
            else ""
        )
        quant_row = f"<dt>quantization</dt><dd>{html.escape(quant.precision)}{detail}</dd>"

    document = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(host.get('cpu', 'unknown'))} · {html.escape(host.get('machine', ''))} ·
generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<p class="sub">Latency measured on this host — median of {final.latency.runs} runs at batch
{final.latency.batch} on {final.latency.threads} thread(s), after {final.latency.warmup} warmup runs.
Accuracy is top-1 of the exported ONNX artifact on the held-out test split.</p>
<div class="cards">{cards}</div>
<h2>Per-stage measurements</h2>
<table><thead><tr><th>stage</th><th>precision</th><th class="num">top-1</th>
<th class="num">&Delta; pp</th><th class="num">median ms</th><th class="num">p95 ms</th>
<th class="num">speedup</th><th class="num">MB</th><th class="num">params</th></tr></thead>
<tbody>{_stage_rows(baseline, result.stages)}</tbody></table>
{charts}
{warnings}
<h2>Reproducibility</h2>
<dl class="kv">
<dt>cpu</dt><dd>{html.escape(host.get('cpu', '?'))} ({host.get('logical_cores', '?')} logical,
{host.get('performance_cores', '?')} performance)</dd>
<dt>ISA features</dt><dd class="mono">{html.escape(', '.join(host.get('features', [])) or 'none detected')}</dd>
<dt>INT8 kernels</dt><dd>{html.escape(host.get('int8_speed', '?'))}</dd>
{quant_row}
<dt>versions</dt><dd class="mono">{html.escape(versions)}</dd>
<dt>final artifact</dt><dd class="mono">{html.escape(final.artifact)}</dd>
</dl>
<footer>Generated by compresskit — latency is measured on this machine, never estimated from FLOPs.</footer>
</div></body></html>"""

    path.write_text(document, encoding="utf-8")
    return path


def render_sweep(baseline: Measurement, rows: list[dict], path: str | Path,
                 title: str = "Compression sweep") -> Path:
    """Report for a ratio x precision sweep: one row per configuration.

    Each row needs 'label' and 'measurement'; the Pareto plot then shows the
    whole front at once so you can pick a trade-off point instead of being
    handed a single black-box answer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = sorted(rows, key=lambda r: -efficiency_score(baseline, r["measurement"]))
    body = "".join(
        f"<tr><td class='mono'>{html.escape(str(r['label']))}</td>"
        f"<td>{html.escape(r['measurement'].precision)}</td>"
        f"<td class='num'>{r['measurement'].accuracy * 100:.2f}%</td>"
        f"<td class='num'>{(baseline.accuracy - r['measurement'].accuracy) * 100:+.2f}</td>"
        f"<td class='num'>{r['measurement'].latency.median_ms:.3f}</td>"
        f"<td class='num'>{baseline.latency.median_ms / max(r['measurement'].latency.median_ms, 1e-9):.2f}x</td>"
        f"<td class='num'>{r['measurement'].size_mb:.2f}</td>"
        f"<td class='num'>{efficiency_score(baseline, r['measurement']):.4f}</td></tr>"
        for r in ordered
    )
    chart = _png_data_uri(pareto_figure([baseline] + [r["measurement"] for r in ordered]))
    host = baseline.host

    document = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(title)}</title><style>{_CSS}</style></head><body><div class="wrap">
<h1>{html.escape(title)}</h1>
<p class="sub">{html.escape(host.get('cpu', '?'))} · {len(ordered)} configurations ·
generated {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
<h2>Configurations, ranked by efficiency score</h2>
<table><thead><tr><th>config</th><th>precision</th><th class="num">top-1</th>
<th class="num">&Delta; pp</th><th class="num">median ms</th><th class="num">speedup</th>
<th class="num">MB</th><th class="num">efficiency</th></tr></thead><tbody>{body}</tbody></table>
<h2>Pareto front</h2><img src="{chart}" alt="pareto">
<footer>Generated by compresskit — latency measured on this machine.</footer>
</div></body></html>"""
    path.write_text(document, encoding="utf-8")
    return path


def render_search(search, path: str | Path, title: str = "Budget search report") -> Path:
    """Report for a SearchResult: every attempt, and why the winner won."""
    path = Path(path)
    if search.chosen is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"<!DOCTYPE html><html><body><h1>{html.escape(title)}</h1>"
            "<p>No candidate was produced.</p></body></html>",
            encoding="utf-8",
        )
        return path

    written = render(search.chosen.result, path, sensitivity=search.sensitivity, title=title)

    rows = "".join(
        f"<tr><td class='mono'>{html.escape(a.label)}</td>"
        f"<td>{'met' if a.met else 'miss'}</td>"
        f"<td class='num'>{a.result.final.accuracy * 100:.2f}%</td>"
        f"<td class='num'>{a.result.final.latency.median_ms:.3f}</td>"
        f"<td class='num'>{a.result.final.size_mb:.2f}</td>"
        f"<td class='num'>{a.score:.4f}</td>"
        f"<td>{html.escape('; '.join(a.failures) or '—')}</td></tr>"
        for a in search.attempts
    )
    block = f"""<h2>Budget search</h2>
<p class="sub">Budget: <span class="mono">{html.escape(search.budget.describe())}</span> —
chose <span class="mono">{html.escape(search.chosen.label)}</span>
({'budget met' if search.met else 'best effort, budget not met'}).</p>
<table><thead><tr><th>attempt</th><th>verdict</th><th class="num">top-1</th>
<th class="num">median ms</th><th class="num">MB</th><th class="num">efficiency</th>
<th>shortfall</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Per-stage measurements</h2>"""

    document = written.read_text(encoding="utf-8").replace("<h2>Per-stage measurements</h2>", block, 1)
    written.write_text(document, encoding="utf-8")
    return written
