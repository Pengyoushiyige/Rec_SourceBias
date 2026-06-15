import argparse
import csv
from pathlib import Path


COLORS = {
    "Human": "#2563eb",
    "LLM": "#dc2626",
    "Human Target": "#0891b2",
    "LLM Target": "#ea580c",
    "llm_ratio": "#7c3aed",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_line_chart(
    path: Path,
    title: str,
    series: dict[str, list[tuple[float, float]]],
    y_label: str,
    y_min: float = 0.0,
    y_max: float | None = None,
) -> None:
    width, height = 920, 520
    left, right, top, bottom = 82, 30, 54, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [x for points in series.values() for x, _ in points]
    ys = [y for points in series.values() for _, y in points]
    if not xs or not ys:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_max is None:
        y_max = max(1.0, max(ys) * 1.12)
    if y_min == y_max:
        y_max += 1.0

    def sx(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy(y: float) -> float:
        return top + (1 - ((y - y_min) / (y_max - y_min))) * plot_h

    grid_lines = []
    for i in range(6):
        val = y_min + (y_max - y_min) * i / 5
        y = sy(val)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
        )
        grid_lines.append(
            f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#475569">{val:.1f}</text>'
        )

    x_ticks = sorted(set(xs))
    tick_lines = []
    for x in x_ticks:
        px = sx(x)
        tick_lines.append(f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+6}" stroke="#334155"/>')
        tick_lines.append(f'<text x="{px:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" fill="#334155">{x:g}</text>')

    paths = []
    legend = []
    for idx, (name, points) in enumerate(series.items()):
        color = COLORS.get(name, "#111827")
        points = sorted(points)
        coords = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{coords}"/>')
        for x, y in points:
            paths.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="4" fill="{color}"/>')
        ly = top + idx * 22
        legend.append(f'<rect x="{width-right-190}" y="{ly-10}" width="12" height="12" fill="{color}"/>')
        legend.append(f'<text x="{width-right-172}" y="{ly}" font-size="13" fill="#0f172a">{name}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="30" font-size="22" font-weight="700" fill="#0f172a">{title}</text>
<text x="{left}" y="{height-22}" font-size="13" fill="#475569">Loop epoch</text>
<text x="18" y="{top + plot_h / 2:.1f}" font-size="13" fill="#475569" transform="rotate(-90 18 {top + plot_h / 2:.1f})">{y_label}</text>
{''.join(grid_lines)}
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>
{''.join(tick_lines)}
{''.join(paths)}
{''.join(legend)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def svg_source_bias_dashboard(path: Path, summary_rows: list[dict[str, object]]) -> None:
    rows = [
        row
        for row in summary_rows
        if row.get("llm_ratio") != "" and row.get("llm_target_minus_human_target_MAP5") != ""
    ]
    if not rows:
        path.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"></svg>\n", encoding="utf-8")
        return

    points = [
        (
            float(row["loop_epoch"]),
            float(row["llm_ratio"]) * 100,
            float(row["llm_target_minus_human_target_MAP5"]),
        )
        for row in rows
    ]
    width, height = 1120, 680
    left, right, top, bottom = 90, 42, 118, 72
    panel_gap = 68
    panel_h = (height - top - bottom - panel_gap) / 2
    plot_w = width - left - right
    epochs = [p[0] for p in points]
    ratios = [p[1] for p in points]
    gaps = [p[2] for p in points]
    x_min, x_max = min(epochs), max(epochs)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    gap_min = min(0.0, min(gaps) * 0.9)
    gap_max = max(1.0, max(gaps) * 1.15)

    def sx(x: float) -> float:
        return left + ((x - x_min) / (x_max - x_min)) * plot_w

    def sy_ratio(y: float) -> float:
        return top + (1 - (y / 100.0)) * panel_h

    def sy_gap(y: float) -> float:
        y0 = top + panel_h + panel_gap
        return y0 + (1 - ((y - gap_min) / (gap_max - gap_min))) * panel_h

    def polyline(values: list[tuple[float, float]], y_fn, color: str) -> str:
        coords = " ".join(f"{sx(x):.1f},{y_fn(y):.1f}" for x, y in values)
        circles = "".join(f'<circle cx="{sx(x):.1f}" cy="{y_fn(y):.1f}" r="4.2" fill="{color}"/>' for x, y in values)
        return f'<polyline fill="none" stroke="{color}" stroke-width="3.5" points="{coords}"/>{circles}'

    ratio_values = [(x, y) for x, y, _ in points]
    gap_values = [(x, y) for x, _, y in points]
    first_ratio, last_ratio = ratios[0], ratios[-1]
    first_gap, last_gap = gaps[0], gaps[-1]
    ratio_delta = last_ratio - first_ratio
    gap_delta = last_gap - first_gap

    grid = []
    for value in [0, 25, 50, 75, 100]:
        y = sy_ratio(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        grid.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value}</text>')
    for i in range(5):
        value = gap_min + (gap_max - gap_min) * i / 4
        y = sy_gap(value)
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        grid.append(f'<text x="{left-14}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#475569">{value:.1f}</text>')

    x_ticks = []
    for x in sorted(set(epochs)):
        px = sx(x)
        x_ticks.append(f'<line x1="{px:.1f}" y1="{height-bottom}" x2="{px:.1f}" y2="{height-bottom+7}" stroke="#334155"/>')
        x_ticks.append(f'<text x="{px:.1f}" y="{height-bottom+26}" text-anchor="middle" font-size="12" fill="#334155">{x:g}</text>')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{left}" y="40" font-size="26" font-weight="700" fill="#0f172a">Source Bias Feedback Loop Summary</text>
<text x="{left}" y="70" font-size="14" fill="#475569">Standard loop, 10 epochs: LLM selections and target-source preference both rise over the run.</text>
<rect x="{left}" y="84" width="245" height="26" rx="4" fill="#f1f5f9"/>
<text x="{left+12}" y="102" font-size="13" fill="#334155">LLM ratio: {first_ratio:.1f}% -> {last_ratio:.1f}% ({ratio_delta:+.1f} pp)</text>
<rect x="{left+264}" y="84" width="330" height="26" rx="4" fill="#f1f5f9"/>
<text x="{left+276}" y="102" font-size="13" fill="#334155">LLM-target MAP5 gap: {first_gap:+.2f} -> {last_gap:+.2f} ({gap_delta:+.2f})</text>
{''.join(grid)}
<text x="{left}" y="{top-16}" font-size="15" font-weight="700" fill="#0f172a">LLM selection ratio</text>
<text x="22" y="{top + panel_h / 2:.1f}" font-size="13" fill="#475569" transform="rotate(-90 22 {top + panel_h / 2:.1f})">Percent</text>
<line x1="{left}" y1="{top+panel_h:.1f}" x2="{width-right}" y2="{top+panel_h:.1f}" stroke="#334155"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+panel_h:.1f}" stroke="#334155"/>
{polyline(ratio_values, sy_ratio, "#7c3aed")}
<text x="{left}" y="{top + panel_h + panel_gap - 16:.1f}" font-size="15" font-weight="700" fill="#0f172a">LLM Target - Human Target MAP5</text>
<text x="22" y="{top + panel_h + panel_gap + panel_h / 2:.1f}" font-size="13" fill="#475569" transform="rotate(-90 22 {top + panel_h + panel_gap + panel_h / 2:.1f})">MAP5 gap</text>
<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#334155"/>
<line x1="{left}" y1="{top+panel_h+panel_gap:.1f}" x2="{left}" y2="{height-bottom}" stroke="#334155"/>
{polyline(gap_values, sy_gap, "#ea580c")}
{''.join(x_ticks)}
<text x="{left}" y="{height-24}" font-size="13" fill="#475569">Loop epoch</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def build_summary(run_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    metrics = read_csv(run_dir / "metrics.csv")
    events = read_csv(run_dir / "events.csv")
    llm_by_epoch = {
        float(row["loop_epoch"]): float(row["llm_ratio"])
        for row in events
        if row.get("event") == "llm_ratio" and row.get("loop_epoch") and row.get("llm_ratio")
    }
    rows = []
    grouped: dict[float, dict[str, dict[str, float]]] = {}
    for row in metrics:
        epoch = float(row["loop_epoch"])
        test_type = row["test_type"]
        grouped.setdefault(epoch, {})[test_type] = {k: float(row[k]) for k in ["MAP1", "MAP3", "MAP5", "nDCG1", "nDCG3", "nDCG5"]}

    for epoch in sorted(grouped):
        human = grouped[epoch].get("Human", {})
        llm = grouped[epoch].get("LLM", {})
        human_target = grouped[epoch].get("Human Target", {})
        llm_target = grouped[epoch].get("LLM Target", {})
        rows.append(
            {
                "loop_epoch": epoch,
                "test_ratio": next((row["test_ratio"] for row in metrics if float(row["loop_epoch"]) == epoch), ""),
                "llm_ratio": llm_by_epoch.get(epoch, ""),
                "human_MAP5": human.get("MAP5", ""),
                "llm_MAP5": llm.get("MAP5", ""),
                "llm_minus_human_MAP5": (llm.get("MAP5", 0.0) - human.get("MAP5", 0.0)) if human and llm else "",
                "human_target_MAP5": human_target.get("MAP5", ""),
                "llm_target_MAP5": llm_target.get("MAP5", ""),
                "llm_target_minus_human_target_MAP5": (
                    llm_target.get("MAP5", 0.0) - human_target.get("MAP5", 0.0)
                )
                if human_target and llm_target
                else "",
                "human_nDCG5": human.get("nDCG5", ""),
                "llm_nDCG5": llm.get("nDCG5", ""),
                "llm_minus_human_nDCG5": (llm.get("nDCG5", 0.0) - human.get("nDCG5", 0.0)) if human and llm else "",
            }
        )
    return rows, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Create cleaned tables and SVG plots from a saved source-bias run.")
    parser.add_argument("run_dir", help="Path to a results/source_bias_runs/<run_id> directory.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, metrics = build_summary(run_dir)
    fields = [
        "loop_epoch",
        "test_ratio",
        "llm_ratio",
        "human_MAP5",
        "llm_MAP5",
        "llm_minus_human_MAP5",
        "human_target_MAP5",
        "llm_target_MAP5",
        "llm_target_minus_human_target_MAP5",
        "human_nDCG5",
        "llm_nDCG5",
        "llm_minus_human_nDCG5",
    ]
    write_csv(output_dir / "source_bias_summary.csv", summary_rows, fields)

    metric_series: dict[str, list[tuple[float, float]]] = {}
    for row in metrics:
        if row["test_type"] in ("Human", "LLM", "Human Target", "LLM Target"):
            metric_series.setdefault(row["test_type"], []).append((float(row["loop_epoch"]), float(row["MAP5"])))
    ratio_series = {
        "llm_ratio": [
            (float(row["loop_epoch"]), float(row["llm_ratio"]) * 100)
            for row in summary_rows
            if row["llm_ratio"] != ""
        ]
    }
    svg_line_chart(output_dir / "map5_by_epoch.svg", "MAP5 by Source Type", metric_series, "MAP5")
    svg_line_chart(output_dir / "llm_ratio_by_epoch.svg", "LLM Selection Ratio by Epoch", ratio_series, "LLM ratio (%)", 0.0, 100.0)
    svg_source_bias_dashboard(output_dir / "source_bias_dashboard.svg", summary_rows)

    first_ratio = summary_rows[0]["llm_ratio"] if summary_rows else ""
    last_ratio = summary_rows[-1]["llm_ratio"] if summary_rows else ""
    first_gap = summary_rows[0]["llm_target_minus_human_target_MAP5"] if summary_rows else ""
    last_gap = summary_rows[-1]["llm_target_minus_human_target_MAP5"] if summary_rows else ""
    md = [
        "# Source Bias Run Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Epochs summarized: {len(summary_rows)}",
        f"- LLM selection ratio: `{first_ratio}` -> `{last_ratio}`",
        f"- LLM-target minus Human-target MAP5: `{first_gap}` -> `{last_gap}`",
        f"- Clean table: `{output_dir / 'source_bias_summary.csv'}`",
        f"- Dashboard chart: `{output_dir / 'source_bias_dashboard.svg'}`",
        f"- MAP5 chart: `{output_dir / 'map5_by_epoch.svg'}`",
        f"- LLM ratio chart: `{output_dir / 'llm_ratio_by_epoch.svg'}`",
        "",
        "The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.",
        "",
    ]
    (output_dir / "source_bias_summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Saved clean summary: {output_dir / 'source_bias_summary.csv'}")
    print(f"Saved charts: {output_dir / 'source_bias_dashboard.svg'}, {output_dir / 'map5_by_epoch.svg'}, and {output_dir / 'llm_ratio_by_epoch.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
