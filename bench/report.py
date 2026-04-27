"""Report emitters: markdown rollup + static HTML dashboard (Chart.js CDN).

Judge modes supported in the rendered output:
  pairwise — NxN round-robin. Per-model tournament win rate
             (wins + 0.5 * ties) / matches. Also renders a win-rate matrix.
  scored   — 1-5 rubric. Per-model mean score column in the rollup.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

# --- Pareto helper ---------------------------------------------------------

def _pareto_frontier(points: list[tuple[str, float, float]]) -> list[str]:
    """Return model ids on the Pareto frontier (quality vs energy).

    Input tuples are (model_id, quality, energy). Higher quality is better;
    lower energy is better. A point is non-dominated iff no other point has
    both quality >= p.quality and energy <= p.energy with at least one
    strict inequality. Output is sorted by ascending energy (the natural
    plotting order).
    """
    frontier = [
        p for p in points
        if not any(
            q is not p and q[1] >= p[1] and q[2] <= p[2]
            and (q[1] > p[1] or q[2] < p[2])
            for q in points
        )
    ]
    return [m for m, _, _ in sorted(frontier, key=lambda p: p[2])]


# --- percentile helpers ----------------------------------------------------

def _percentile(xs: list[float], p: float) -> float | None:
    """Linear-interpolation percentile. p in [0, 100]. Returns None for [].

    Matches numpy's default (method="linear") so cross-checking against a
    pandas/numpy rollup during debugging doesn't produce spurious deltas.
    """
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


# --- numeric rollup from records -------------------------------------------


def _fmt(v: float | None, nd: int = 3) -> str:
    """Format float to N decimal places, or em-dash if None."""
    return "—" if v is None else f"{v:.{nd}f}"


def _extract_field(rows: list[dict[str, Any]], key: str) -> list[float]:
    """Extract non-None values for a key from a list of dicts."""
    return [r[key] for r in rows if r.get(key) is not None]


def _rollup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("error"):
            continue
        by_model[r["model_id"]].append(r)

    out: dict[str, dict[str, Any]] = {}
    for mid, rows in by_model.items():
        lat = _extract_field(rows, "latency_s")
        ttft = _extract_field(rows, "ttft_s")
        tps = _extract_field(rows, "tokens_per_sec")
        wh = _extract_field(rows, "energy_wh")
        usd = _extract_field(rows, "cost_usd")
        qh = _extract_field(rows, "quality_heuristic")
        out[mid] = {
            "n": len(rows),
            "latency_mean_s": mean(lat) if lat else None,
            "latency_p50_s": _percentile(lat, 50),
            "latency_p95_s": _percentile(lat, 95),
            "latency_p99_s": _percentile(lat, 99),
            "ttft_mean_s": mean(ttft) if ttft else None,
            "ttft_p50_s": _percentile(ttft, 50),
            "ttft_p95_s": _percentile(ttft, 95),
            "ttft_p99_s": _percentile(ttft, 99),
            "tokens_per_sec_mean": mean(tps) if tps else None,
            "energy_wh_total": sum(wh) if wh else None,
            "cost_usd_total": sum(usd) if usd else None,
            "quality_heuristic_mean": mean(qh) if qh else None,
        }
    return out


# --- judge rollups ---------------------------------------------------------

def _detect_mode(judgements: list[dict[str, Any]]) -> str | None:
    if not judgements:
        return None
    # Prefer explicit tag; fall back to shape inference for legacy payloads.
    for j in judgements:
        m = j.get("mode")
        if m in {"pairwise", "scored"}:
            return m
    if any("score" in j for j in judgements):
        return "scored"
    return "pairwise"


def _pairwise_rollup(judgements: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model W/L/T aggregated across every pair the model appeared in,
    plus an NxN win-rate matrix (tournament win rate, ties = 0.5).
    """
    per_model: dict[str, dict[str, int]] = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})
    # matrix[row][col] = aggregate over matches where row vs col:
    #   numerator = wins(row) + 0.5 * ties
    #   denominator = total matches
    mat_num: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    mat_den: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for j in judgements:
        a, b = j.get("a_model"), j.get("b_model")
        w = j.get("winner")
        if not a or not b or w is None:
            continue
        mat_den[a][b] += 1
        mat_den[b][a] += 1
        if w == "A":
            per_model[a]["w"] += 1
            per_model[b]["l"] += 1
            mat_num[a][b] += 1.0
        elif w == "B":
            per_model[b]["w"] += 1
            per_model[a]["l"] += 1
            mat_num[b][a] += 1.0
        else:  # TIE
            per_model[a]["t"] += 1
            per_model[b]["t"] += 1
            mat_num[a][b] += 0.5
            mat_num[b][a] += 0.5

    models = sorted(per_model.keys())
    rates: dict[str, float | None] = {}
    for m in models:
        stats = per_model[m]
        n = stats["w"] + stats["l"] + stats["t"]
        rates[m] = ((stats["w"] + 0.5 * stats["t"]) / n) if n else None

    matrix: dict[str, dict[str, float | None]] = {}
    for row in models:
        matrix[row] = {}
        for col in models:
            if row == col:
                matrix[row][col] = None
                continue
            d = mat_den[row][col]
            matrix[row][col] = (mat_num[row][col] / d) if d else None

    return {
        "models": models,
        "per_model": {m: {**per_model[m], "win_rate": rates[m]} for m in models},
        "matrix": matrix,
    }


def _scored_rollup(judgements: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[int]] = defaultdict(list)
    for j in judgements:
        s = j.get("score")
        mid = j.get("model_id")
        if s is None or mid is None:
            continue
        by_model[mid].append(int(s))
    out: dict[str, dict[str, Any]] = {}
    for m, scores in by_model.items():
        out[m] = {
            "n": len(scores),
            "mean": mean(scores) if scores else None,
            "stdev": pstdev(scores) if len(scores) > 1 else 0.0,
        }
    return {"models": sorted(out.keys()), "per_model": out}


# --- markdown --------------------------------------------------------------

def _fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def emit_markdown(payload: dict[str, Any], path: Path) -> None:
    roll = _rollup(payload["records"])
    judgements = payload.get("judgements") or []
    mode = _detect_mode(judgements)

    lines: list[str] = []
    lines.append(f"# Benchmark: {payload.get('run_id', payload['timestamp'])}")
    lines.append("")
    lines.append(f"Timestamp: `{payload['timestamp']}`")
    if mode:
        lines.append(f"Judge mode: `{mode}`")
    prov = payload.get("provenance") or {}
    if prov:
        git = prov.get("git") or {}
        parts = []
        if git.get("sha"):
            dirty = " (dirty)" if git.get("dirty") else ""
            parts.append(f"git `{git['sha']}{dirty}`")
        if prov.get("config_hash"):
            parts.append(f"config `{prov['config_hash']}`")
        if prov.get("seed") is not None:
            parts.append(f"seed `{prov['seed']}`")
        if prov.get("llama_swap_version"):
            parts.append(f"llama-swap `{prov['llama_swap_version']}`")
        if prov.get("server_image"):
            parts.append(f"image `{prov['server_image']}`")
        if prov.get("python"):
            py = prov["python"].split()[0]
            parts.append(f"Python `{py}`")
        if parts:
            lines.append(" · ".join(parts))
    lines.append("")
    lines.append("## Per-model rollup")
    lines.append("")

    if mode == "scored":
        sr = _scored_rollup(judgements)
        lines.append("| Model | N | Latency mean (s) | Tok/sec mean | Energy (Wh) | Cost (USD) | Heuristic | Judge score (mean ± sd) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for mid, r in roll.items():
            s = sr["per_model"].get(mid)
            score_cell = (
                f"{_fmt(s['mean'], 2)} ± {_fmt(s['stdev'], 2)} (n={s['n']})"
                if s else "—"
            )
            lines.append(
                f"| {mid} | {r['n']} | {_fmt(r['latency_mean_s'])} | "
                f"{_fmt(r['tokens_per_sec_mean'], 2)} | {_fmt(r['energy_wh_total'])} | "
                f"{_fmt(r['cost_usd_total'], 4)} | {_fmt(r['quality_heuristic_mean'], 3)} | "
                f"{score_cell} |"
            )
    else:
        # pairwise (or no judge)
        lines.append("| Model | N | Latency mean (s) | Tok/sec mean | Energy (Wh) | Cost (USD) | Heuristic |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for mid, r in roll.items():
            lines.append(
                f"| {mid} | {r['n']} | {_fmt(r['latency_mean_s'])} | "
                f"{_fmt(r['tokens_per_sec_mean'], 2)} | {_fmt(r['energy_wh_total'])} | "
                f"{_fmt(r['cost_usd_total'], 4)} | {_fmt(r['quality_heuristic_mean'], 3)} |"
            )

    lines.append("")

    # Latency distribution + TTFT: a separate section keeps the rollup
    # table narrow while still surfacing the tails that mean hides.
    lines.append("## Latency distribution (s)")
    lines.append("")
    lines.append("| Model | mean | p50 | p95 | p99 |")
    lines.append("|---|---:|---:|---:|---:|")
    for mid, r in roll.items():
        lines.append(
            f"| {mid} | {_fmt(r['latency_mean_s'])} | "
            f"{_fmt(r['latency_p50_s'])} | {_fmt(r['latency_p95_s'])} | "
            f"{_fmt(r['latency_p99_s'])} |"
        )
    lines.append("")

    # TTFT may be all-None on legacy runs that predate streaming; skip then.
    if any(r.get("ttft_mean_s") is not None for r in roll.values()):
        lines.append("## Time-to-first-token (s)")
        lines.append("")
        lines.append("| Model | mean | p50 | p95 | p99 |")
        lines.append("|---|---:|---:|---:|---:|")
        for mid, r in roll.items():
            lines.append(
                f"| {mid} | {_fmt(r['ttft_mean_s'])} | "
                f"{_fmt(r['ttft_p50_s'])} | {_fmt(r['ttft_p95_s'])} | "
                f"{_fmt(r['ttft_p99_s'])} |"
            )
        lines.append("")

    if mode == "pairwise":
        pr = _pairwise_rollup(judgements)
        lines.append("## Tournament (pairwise)")
        lines.append("")
        lines.append("| Model | W | L | T | Win rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for m in pr["models"]:
            s = pr["per_model"][m]
            wr = s["win_rate"]
            lines.append(
                f"| {m} | {s['w']} | {s['l']} | {s['t']} | "
                f"{_fmt(wr * 100 if wr is not None else None, 1)}% |"
            )
        lines.append("")
        if len(pr["models"]) >= 2:
            lines.append("### Win-rate matrix (row vs column)")
            lines.append("")
            header = "| vs | " + " | ".join(pr["models"]) + " |"
            sep = "|---|" + "---:|" * len(pr["models"])
            lines.append(header)
            lines.append(sep)
            for row in pr["models"]:
                cells = []
                for col in pr["models"]:
                    v = pr["matrix"][row][col]
                    cells.append("—" if v is None else f"{v*100:.0f}%")
                lines.append(f"| **{row}** | " + " | ".join(cells) + " |")
    elif mode == "scored":
        lines.append("## Scored rubric (1-5)")
        lines.append("")
        lines.append("_Per-model mean score shown in the rollup above. "
                     "Higher = judge preferred the response against an absolute rubric._")
    else:
        lines.append("## Judge")
        lines.append("")
        lines.append("_Judge disabled or no judgements produced._")

    # Pareto frontier: quality vs total energy. Use the richest quality
    # signal available for the run; fall back to heuristic.
    pts: list[tuple[str, float, float]] = []
    if mode == "pairwise":
        pr = _pairwise_rollup(judgements)
        for m, v in pr["per_model"].items():
            q = v.get("win_rate")
            e = roll.get(m, {}).get("energy_wh_total")
            if q is not None and e is not None:
                pts.append((m, float(q), float(e)))
        q_label = "win rate"
    elif mode == "scored":
        sr = _scored_rollup(judgements)
        for m, v in sr["per_model"].items():
            q = v.get("mean")
            e = roll.get(m, {}).get("energy_wh_total")
            if q is not None and e is not None:
                pts.append((m, float(q) / 5.0, float(e)))
        q_label = "judge score (mean/5)"
    else:
        for m, r in roll.items():
            q = r.get("quality_heuristic_mean")
            e = r.get("energy_wh_total")
            if q is not None and e is not None:
                pts.append((m, float(q), float(e)))
        q_label = "heuristic quality"

    if len(pts) >= 2:
        frontier = _pareto_frontier(pts)
        lines.append("")
        lines.append("## Pareto: quality vs energy")
        lines.append("")
        lines.append(f"Quality axis: _{q_label}_ (higher better). "
                     "Energy axis: total Wh (lower better).")
        lines.append("")
        lines.append("**Frontier (left to right, ascending energy):** "
                     + " → ".join(frontier))

    path.write_text("\n".join(lines) + "\n")


# --- HTML ------------------------------------------------------------------

_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Benchmark {run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}}
h1{{margin-bottom:.25rem}}
.meta{{color:#666;margin-bottom:1.5rem}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
.card{{border:1px solid #ddd;border-radius:8px;padding:1rem}}
table{{border-collapse:collapse;width:100%}}
th,td{{border-bottom:1px solid #eee;padding:.4rem .6rem;text-align:right}}
th:first-child,td:first-child{{text-align:left}}
.matrix td{{text-align:center;font-variant-numeric:tabular-nums}}
.matrix th{{text-align:center}}
.cell-win{{background:#d4ecd4}}
.cell-loss{{background:#ecd4d4}}
.cell-draw{{background:#f0f0f0}}
</style></head><body>
<h1>Benchmark: {run_id}</h1>
<div class="meta">Timestamp: {ts} · Judge mode: <code id="mode"></code>{prov_line}</div>

<div class="card">
  <h2>Per-model rollup</h2>
  <table id="rollup"></table>
</div>

<div class="card" style="margin-top:1.5rem">
  <h2>Latency distribution (s)</h2>
  <table id="lat_dist"></table>
</div>

<div class="card" style="margin-top:1.5rem" id="ttft_card" hidden>
  <h2>Time-to-first-token (s)</h2>
  <table id="ttft_dist"></table>
</div>

<div class="grid" style="margin-top:1.5rem">
  <div class="card"><h3>Mean latency (s)</h3><canvas id="c_lat"></canvas></div>
  <div class="card"><h3>Mean tokens/sec</h3><canvas id="c_tps"></canvas></div>
  <div class="card"><h3>Total energy (Wh)</h3><canvas id="c_wh"></canvas></div>
  <div class="card"><h3>Total cost (USD)</h3><canvas id="c_usd"></canvas></div>
</div>

<div class="card" style="margin-top:1.5rem">
  <h2 id="judge-h">Judge</h2>
  <div id="judge-summary"></div>
  <div id="judge-matrix-wrap"></div>
  <canvas id="c_judge" style="max-height:260px;margin-top:1rem"></canvas>
</div>

<div class="card" style="margin-top:1.5rem" id="pareto_card" hidden>
  <h2>Pareto: quality vs energy</h2>
  <div id="pareto_summary" style="color:#666"></div>
  <canvas id="c_pareto" style="max-height:420px;margin-top:.5rem"></canvas>
</div>

<script>
const data = {payload_json};

function detectMode(js) {{
  if (!js || !js.length) return null;
  for (const j of js) if (j.mode === "pairwise" || j.mode === "scored") return j.mode;
  if (js.some(j => "score" in j)) return "scored";
  return "pairwise";
}}

const mode = detectMode(data.judgements);
document.getElementById("mode").textContent = mode || "none";

function percentile(xs, p) {{
  if (!xs.length) return null;
  const s = [...xs].sort((a,b)=>a-b);
  if (s.length === 1) return s[0];
  const k = (s.length - 1) * (p / 100);
  const lo = Math.floor(k);
  const hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
}}

function rollup(records) {{
  const by = {{}};
  for (const r of records) {{
    if (r.error) continue;
    const k = r.model_id;
    by[k] ??= {{n:0, lat:[], ttft:[], tps:[], wh:[], usd:[], qh:[]}};
    by[k].n++;
    if (r.latency_s != null) by[k].lat.push(r.latency_s);
    if (r.ttft_s != null) by[k].ttft.push(r.ttft_s);
    if (r.tokens_per_sec) by[k].tps.push(r.tokens_per_sec);
    if (r.energy_wh != null) by[k].wh.push(r.energy_wh);
    if (r.cost_usd != null) by[k].usd.push(r.cost_usd);
    if (r.quality_heuristic != null) by[k].qh.push(r.quality_heuristic);
  }}
  const mean = a => a.length ? a.reduce((s,x)=>s+x,0)/a.length : null;
  const sum  = a => a.length ? a.reduce((s,x)=>s+x,0) : null;
  const out = {{}};
  for (const [k,v] of Object.entries(by)) {{
    out[k] = {{
      n: v.n,
      latency_mean_s: mean(v.lat),
      latency_p50_s: percentile(v.lat, 50),
      latency_p95_s: percentile(v.lat, 95),
      latency_p99_s: percentile(v.lat, 99),
      ttft_mean_s: mean(v.ttft),
      ttft_p50_s: percentile(v.ttft, 50),
      ttft_p95_s: percentile(v.ttft, 95),
      ttft_p99_s: percentile(v.ttft, 99),
      tokens_per_sec_mean: mean(v.tps),
      energy_wh_total: sum(v.wh),
      cost_usd_total: sum(v.usd),
      quality_heuristic_mean: mean(v.qh),
    }};
  }}
  return out;
}}

function scoredRollup(js) {{
  const by = {{}};
  for (const j of js) {{
    if (j.score == null || !j.model_id) continue;
    (by[j.model_id] ??= []).push(j.score);
  }}
  const out = {{}};
  for (const [m, arr] of Object.entries(by)) {{
    const mean = arr.reduce((s,x)=>s+x,0)/arr.length;
    const variance = arr.length > 1
      ? arr.reduce((s,x)=>s+(x-mean)**2,0)/arr.length : 0;
    out[m] = {{n: arr.length, mean, stdev: Math.sqrt(variance)}};
  }}
  return out;
}}

function pairwiseRollup(js) {{
  const per = {{}};
  const num = {{}}, den = {{}};
  const touch = (a, b) => {{
    per[a] ??= {{w:0, l:0, t:0}};
    per[b] ??= {{w:0, l:0, t:0}};
    num[a] ??= {{}}; num[b] ??= {{}};
    den[a] ??= {{}}; den[b] ??= {{}};
    num[a][b] ??= 0; num[b][a] ??= 0;
    den[a][b] ??= 0; den[b][a] ??= 0;
  }};
  for (const j of js) {{
    const a = j.a_model, b = j.b_model, w = j.winner;
    if (!a || !b || w == null) continue;
    touch(a, b);
    den[a][b]++; den[b][a]++;
    if (w === "A") {{ per[a].w++; per[b].l++; num[a][b] += 1; }}
    else if (w === "B") {{ per[b].w++; per[a].l++; num[b][a] += 1; }}
    else {{ per[a].t++; per[b].t++; num[a][b] += 0.5; num[b][a] += 0.5; }}
  }}
  const models = Object.keys(per).sort();
  const rates = {{}};
  for (const m of models) {{
    const s = per[m];
    const n = s.w + s.l + s.t;
    rates[m] = n ? (s.w + 0.5 * s.t) / n : null;
  }}
  const matrix = {{}};
  for (const row of models) {{
    matrix[row] = {{}};
    for (const col of models) {{
      if (row === col) {{ matrix[row][col] = null; continue; }}
      const d = den[row]?.[col] ?? 0;
      matrix[row][col] = d ? (num[row]?.[col] ?? 0) / d : null;
    }}
  }}
  return {{models, per, rates, matrix}};
}}

const roll = rollup(data.records);
const models = Object.keys(roll);
const fmt = (v, nd=3) => v==null ? "—" : Number(v).toFixed(nd);
const fmtPct = v => v==null ? "—" : (v*100).toFixed(0) + "%";

// Rollup table
(() => {{
  const t = document.getElementById("rollup");
  let header = "<tr><th>Model</th><th>N</th><th>Latency mean (s)</th><th>Tok/sec mean</th><th>Energy (Wh)</th><th>Cost (USD)</th><th>Heur quality</th>";
  if (mode === "scored") header += "<th>Judge score</th>";
  if (mode === "pairwise") header += "<th>Win rate</th>";
  header += "</tr>";

  let scored = null, pair = null;
  if (mode === "scored") scored = scoredRollup(data.judgements);
  if (mode === "pairwise") pair = pairwiseRollup(data.judgements);

  t.innerHTML = header + models.map(m => {{
    const r = roll[m];
    let extra = "";
    if (mode === "scored") {{
      const s = scored[m];
      extra = "<td>" + (s ? (fmt(s.mean,2) + " ± " + fmt(s.stdev,2) + " (n=" + s.n + ")") : "—") + "</td>";
    }} else if (mode === "pairwise") {{
      extra = "<td>" + (pair.rates[m] != null ? fmtPct(pair.rates[m]) : "—") + "</td>";
    }}
    return `<tr><td>${{m}}</td><td>${{r.n}}</td><td>${{fmt(r.latency_mean_s)}}</td><td>${{fmt(r.tokens_per_sec_mean,2)}}</td><td>${{fmt(r.energy_wh_total)}}</td><td>${{fmt(r.cost_usd_total,4)}}</td><td>${{fmt(r.quality_heuristic_mean)}}</td>${{extra}}</tr>`;
  }}).join("");
}})();

function bar(id, label, values) {{
  new Chart(document.getElementById(id), {{
    type: "bar",
    data: {{ labels: models, datasets: [{{ label, data: values }}] }},
    options: {{ responsive: true, plugins: {{ legend: {{ display: false }} }} }},
  }});
}}
bar("c_lat", "s",    models.map(m => roll[m].latency_mean_s ?? 0));
bar("c_tps", "tok/s", models.map(m => roll[m].tokens_per_sec_mean ?? 0));
bar("c_wh",  "Wh",    models.map(m => roll[m].energy_wh_total ?? 0));
bar("c_usd", "USD",   models.map(m => roll[m].cost_usd_total ?? 0));

// Latency distribution table
(() => {{
  const t = document.getElementById("lat_dist");
  const header = "<tr><th>Model</th><th>mean</th><th>p50</th><th>p95</th><th>p99</th></tr>";
  t.innerHTML = header + models.map(m => {{
    const r = roll[m];
    return `<tr><td>${{m}}</td><td>${{fmt(r.latency_mean_s)}}</td><td>${{fmt(r.latency_p50_s)}}</td><td>${{fmt(r.latency_p95_s)}}</td><td>${{fmt(r.latency_p99_s)}}</td></tr>`;
  }}).join("");
}})();

// TTFT distribution table — hidden if no streamed calls produced data.
(() => {{
  const anyTtft = models.some(m => roll[m].ttft_mean_s != null);
  if (!anyTtft) return;
  const card = document.getElementById("ttft_card");
  card.hidden = false;
  const t = document.getElementById("ttft_dist");
  const header = "<tr><th>Model</th><th>mean</th><th>p50</th><th>p95</th><th>p99</th></tr>";
  t.innerHTML = header + models.map(m => {{
    const r = roll[m];
    return `<tr><td>${{m}}</td><td>${{fmt(r.ttft_mean_s)}}</td><td>${{fmt(r.ttft_p50_s)}}</td><td>${{fmt(r.ttft_p95_s)}}</td><td>${{fmt(r.ttft_p99_s)}}</td></tr>`;
  }}).join("");
}})();

// Judge section
(() => {{
  const summary = document.getElementById("judge-summary");
  const matrixWrap = document.getElementById("judge-matrix-wrap");
  const canvas = document.getElementById("c_judge");
  const hdr = document.getElementById("judge-h");

  if (!mode || !data.judgements || !data.judgements.length) {{
    hdr.textContent = "Judge";
    summary.textContent = "Judge disabled or no judgements produced.";
    return;
  }}

  if (mode === "scored") {{
    hdr.textContent = "Scored rubric (1-5)";
    const sr = scoredRollup(data.judgements);
    const ordered = models.filter(m => sr[m]);
    summary.innerHTML = "Mean judge score per model (higher is better).";
    new Chart(canvas, {{
      type: "bar",
      data: {{
        labels: ordered,
        datasets: [{{ label: "mean score", data: ordered.map(m => sr[m].mean) }}],
      }},
      options: {{
        responsive: true, plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ min: 1, max: 5 }} }},
      }},
    }});
    return;
  }}

  // pairwise
  hdr.textContent = "Tournament (pairwise)";
  const pr = pairwiseRollup(data.judgements);

  summary.innerHTML = "Win rate = (wins + 0.5 × ties) / matches. Order was randomized per call; swapped verdicts were inverted before counting.";

  // Matrix
  if (pr.models.length >= 2) {{
    let html = '<h3>Win-rate matrix (row vs column)</h3><table class="matrix"><tr><th>vs</th>';
    for (const c of pr.models) html += `<th>${{c}}</th>`;
    html += "</tr>";
    for (const row of pr.models) {{
      html += `<tr><th>${{row}}</th>`;
      for (const col of pr.models) {{
        const v = pr.matrix[row][col];
        if (v == null) {{ html += '<td>—</td>'; continue; }}
        const cls = v > 0.55 ? "cell-win" : v < 0.45 ? "cell-loss" : "cell-draw";
        html += `<td class="${{cls}}">${{(v*100).toFixed(0)}}%</td>`;
      }}
      html += "</tr>";
    }}
    html += "</table>";
    matrixWrap.innerHTML = html;
  }}

  // Overall win rate bar chart
  new Chart(canvas, {{
    type: "bar",
    data: {{
      labels: pr.models,
      datasets: [{{ label: "win rate", data: pr.models.map(m => (pr.rates[m] ?? 0) * 100) }}],
    }},
    options: {{
      responsive: true, plugins: {{ legend: {{ display: false }} }},
      scales: {{ y: {{ min: 0, max: 100, ticks: {{ callback: v => v + "%" }} }} }},
    }},
  }});
}})();

// Pareto: quality vs energy. Quality source depends on available signal:
//   1. Pairwise win rate (if judge mode == pairwise and we have judgements).
//   2. Scored judge mean / 5 (if judge mode == scored).
//   3. Fallback to quality_heuristic_mean from records.
// Lower energy is better (x-axis); higher quality is better (y-axis). The
// Pareto frontier = models not dominated on both axes.
(() => {{
  const card = document.getElementById("pareto_card");
  const canvas = document.getElementById("c_pareto");
  const summary = document.getElementById("pareto_summary");

  let qualitySrc = "heuristic";
  let qualityMap = {{}};
  let qualityLabel = "heuristic quality (0-1)";

  if (mode === "pairwise" && data.judgements && data.judgements.length) {{
    const pr = pairwiseRollup(data.judgements);
    qualityMap = pr.rates;
    qualitySrc = "pairwise";
    qualityLabel = "win rate";
  }} else if (mode === "scored" && data.judgements && data.judgements.length) {{
    const sr = scoredRollup(data.judgements);
    qualityMap = {{}};
    for (const [m, v] of Object.entries(sr)) qualityMap[m] = v.mean / 5;
    qualitySrc = "scored";
    qualityLabel = "judge score (mean / 5)";
  }} else {{
    for (const m of models) qualityMap[m] = roll[m].quality_heuristic_mean;
  }}

  // Only plot models with both axes defined.
  const pts = models
    .map(m => ({{m, q: qualityMap[m], e: roll[m].energy_wh_total}}))
    .filter(p => p.q != null && p.e != null);
  if (pts.length < 2) {{
    // One (or zero) plottable model — Pareto is meaningless. Skip entirely.
    return;
  }}
  card.hidden = false;

  // Pareto frontier: a point is non-dominated when no other point has
  // higher-or-equal quality AND lower-or-equal energy with at least one
  // strict inequality.
  const frontier = pts.filter(p => !pts.some(q =>
    q !== p && q.q >= p.q && q.e <= p.e && (q.q > p.q || q.e < p.e)
  )).sort((a, b) => a.e - b.e);
  const frontierSet = new Set(frontier.map(p => p.m));
  const dominated = pts.filter(p => !frontierSet.has(p.m));

  summary.textContent =
    `Quality source: ${{qualitySrc}}. Frontier (top-left = better): ` +
    frontier.map(p => p.m).join(" → ");

  new Chart(canvas, {{
    type: "scatter",
    data: {{
      datasets: [
        {{
          label: "Pareto frontier",
          data: frontier.map(p => ({{x: p.e, y: p.q, label: p.m}})),
          backgroundColor: "#2a7",
          borderColor: "#2a7",
          showLine: true,
          borderDash: [6, 4],
          pointRadius: 7,
        }},
        {{
          label: "dominated",
          data: dominated.map(p => ({{x: p.e, y: p.q, label: p.m}})),
          backgroundColor: "#c55",
          borderColor: "#c55",
          showLine: false,
          pointRadius: 6,
        }},
      ],
    }},
    options: {{
      responsive: true,
      plugins: {{
        tooltip: {{
          callbacks: {{
            label: ctx => {{
              const d = ctx.raw;
              return `${{d.label}}: ${{qualityLabel}}=${{fmt(d.y)}}, energy=${{fmt(d.x)}} Wh`;
            }},
          }},
        }},
      }},
      scales: {{
        x: {{ title: {{display: true, text: "Total energy (Wh) — lower is better"}} }},
        y: {{ title: {{display: true, text: qualityLabel + " — higher is better"}} }},
      }},
    }},
  }});
}})();
</script>
</body></html>
"""


def _html_prov_line(prov: dict[str, Any]) -> str:
    if not prov:
        return ""
    git = prov.get("git") or {}
    parts = []
    if git.get("sha"):
        dirty = " (dirty)" if git.get("dirty") else ""
        parts.append(f"git <code>{git['sha']}{dirty}</code>")
    if prov.get("config_hash"):
        parts.append(f"config <code>{prov['config_hash']}</code>")
    if prov.get("seed") is not None:
        parts.append(f"seed <code>{prov['seed']}</code>")
    if prov.get("llama_swap_version"):
        parts.append(f"llama-swap <code>{prov['llama_swap_version']}</code>")
    if not parts:
        return ""
    return " · " + " · ".join(parts)


def emit_html(payload: dict[str, Any], path: Path) -> None:
    prov = payload.get("provenance") or {}
    html = _HTML_TEMPLATE.format(
        run_id=payload.get("run_id", payload["timestamp"]),
        ts=payload["timestamp"],
        prov_line=_html_prov_line(prov),
        payload_json=json.dumps(payload),
    )
    path.write_text(html)


def emit_reports(payload: dict[str, Any], out_dir: Path, ts: str, md: bool, html: bool) -> None:
    if md:
        emit_markdown(payload, out_dir / f"run-{ts}.md")
    if html:
        emit_html(payload, out_dir / f"run-{ts}.html")
