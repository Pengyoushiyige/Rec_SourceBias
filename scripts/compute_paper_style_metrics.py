from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METRIC_COLS = ["MAP1", "MAP3", "MAP5", "nDCG1", "nDCG3", "nDCG5"]
TEST_TYPES = ["Human", "LLM", "Human Target", "LLM Target"]

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_csv_if_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> bool:
    """Write a meaningful table and remove a stale header-only copy."""
    if not rows:
        if path.exists():
            path.unlink()
        return False
    write_csv(path, rows, fields)
    return True


def as_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def relative_delta(hgc: float | None, aigc: float | None) -> float | None:
    if hgc is None or aigc is None:
        return None
    denominator = (hgc + aigc) / 2.0
    if denominator == 0:
        return None
    return (hgc - aigc) / denominator * 100.0


def fmt(value: object, digits: int = 6) -> object:
    number = as_float(value)
    if number is None:
        return ""
    return round(number, digits)


def load_summary(run_dir: Path) -> dict[str, object]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {"run_id": run_dir.name, "run_dir": str(run_dir)}
    return json.loads(path.read_text(encoding="utf-8"))


def summary_mode(summary: dict[str, object]) -> str:
    mode = summary.get("mode")
    if mode in {"train", "test", "loop"}:
        return str(mode)
    command = summary.get("command")
    if isinstance(command, list):
        try:
            candidate = command[command.index("--mode") + 1]
        except (ValueError, IndexError):
            candidate = None
        if candidate in {"train", "test", "loop"}:
            return str(candidate)
    return "unknown"


def x_key(row: dict[str, str]) -> tuple[str, str]:
    loop_epoch = row.get("loop_epoch", "")
    test_ratio = row.get("test_ratio", "")
    if loop_epoch not in ("", None):
        return ("loop_epoch", str(loop_epoch))
    return ("test_ratio", str(test_ratio))


def x_value(row: dict[str, str]) -> float | None:
    key, value = x_key(row)
    parsed = as_float(value)
    if parsed is not None:
        return parsed
    if key == "loop_epoch":
        return as_float(row.get("test_ratio"))
    return None


def derive_metric_tables(metrics: list[dict[str, str]], events: list[dict[str, str]]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in metrics:
        test_type = row.get("test_type", "")
        if test_type not in TEST_TYPES:
            continue
        grouped.setdefault(x_key(row), {})[test_type] = row

    llm_ratio_by_epoch = {
        str(row.get("loop_epoch")): as_float(row.get("llm_ratio"))
        for row in events
        if row.get("event") == "llm_ratio" and row.get("loop_epoch") not in ("", None)
    }
    new_ratio_by_epoch = {
        str(row.get("loop_epoch")): as_float(row.get("new_ratio"))
        for row in events
        if row.get("event") == "new_ratio" and row.get("loop_epoch") not in ("", None)
    }

    wide_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for (axis_name, axis_raw), by_type in sorted(grouped.items(), key=lambda item: as_float(item[0][1]) or 0.0):
        representative = next(iter(by_type.values()))
        base: dict[str, object] = {
            "axis": axis_name,
            "axis_value": fmt(axis_raw),
            "loop_epoch": representative.get("loop_epoch", ""),
            "test_ratio": representative.get("test_ratio", ""),
            "llm_ratio": llm_ratio_by_epoch.get(str(representative.get("loop_epoch")), ""),
            "new_ratio": new_ratio_by_epoch.get(str(representative.get("loop_epoch")), ""),
        }
        for metric in METRIC_COLS:
            human = as_float(by_type.get("Human", {}).get(metric))
            llm = as_float(by_type.get("LLM", {}).get(metric))
            human_target = as_float(by_type.get("Human Target", {}).get(metric))
            llm_target = as_float(by_type.get("LLM Target", {}).get(metric))
            target_delta = relative_delta(human_target, llm_target)
            direct_delta = relative_delta(human, llm)

            base[f"{metric}_human"] = fmt(human)
            base[f"{metric}_llm"] = fmt(llm)
            base[f"{metric}_human_target"] = fmt(human_target)
            base[f"{metric}_llm_target"] = fmt(llm_target)
            base[f"{metric}_target_relative_delta"] = fmt(target_delta)
            base[f"{metric}_target_relative_delta_pct"] = fmt(target_delta)
            base[f"{metric}_target_relative_delta_decimal"] = fmt((target_delta / 100.0) if target_delta is not None else None)
            base[f"{metric}_target_llm_minus_human"] = fmt(
                (llm_target - human_target) if human_target is not None and llm_target is not None else None
            )
            base[f"{metric}_diagnostic_direct_relative_delta"] = fmt(direct_delta)
            base[f"{metric}_diagnostic_direct_llm_minus_human"] = fmt(
                (llm - human) if human is not None and llm is not None else None
            )
            long_rows.append(
                {
                    **{k: base[k] for k in ["axis", "axis_value", "loop_epoch", "test_ratio", "llm_ratio", "new_ratio"]},
                    "metric": metric,
                    "target_human": fmt(human_target),
                    "target_llm": fmt(llm_target),
                    "target_relative_delta_pct": fmt(target_delta),
                    "target_relative_delta_decimal": fmt(
                        (target_delta / 100.0) if target_delta is not None else None
                    ),
                    "target_llm_minus_human": base[f"{metric}_target_llm_minus_human"],
                    "direct_human": fmt(human),
                    "direct_llm": fmt(llm),
                    "diagnostic_direct_relative_delta_pct": fmt(direct_delta),
                    "diagnostic_direct_llm_minus_human": base[f"{metric}_diagnostic_direct_llm_minus_human"],
                }
            )
        wide_rows.append(base)

    train_loss = [
        {
            "loop_epoch": row.get("loop_epoch", ""),
            "epoch": row.get("epoch", ""),
            "loss": fmt(row.get("loss")),
        }
        for row in events
        if row.get("event") == "train_loss"
    ]
    return {
        "wide": wide_rows,
        "long": long_rows,
        "train_loss": train_loss,
    }


def summarize(wide_rows: list[dict[str, object]], summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if wide_rows:
        for metric in ["MAP5", "nDCG5"]:
            values = [as_float(row.get(f"{metric}_target_relative_delta")) for row in wide_rows]
            values = [value for value in values if value is not None]
            gaps = [as_float(row.get(f"{metric}_target_llm_minus_human")) for row in wide_rows]
            gaps = [value for value in gaps if value is not None]
            rows.append(
                {
                    "run_id": summary.get("run_id", ""),
                    "metric": metric,
                    "points": len(values),
                    "start_relative_delta": fmt(values[0] if values else None),
                    "end_relative_delta": fmt(values[-1] if values else None),
                    "min_relative_delta": fmt(min(values) if values else None),
                    "max_relative_delta": fmt(max(values) if values else None),
                    "mean_relative_delta": fmt(sum(values) / len(values) if values else None),
                    "start_llm_minus_human": fmt(gaps[0] if gaps else None),
                    "end_llm_minus_human": fmt(gaps[-1] if gaps else None),
                    "min_llm_minus_human": fmt(min(gaps) if gaps else None),
                    "max_llm_minus_human": fmt(max(gaps) if gaps else None),
                }
            )
    return rows


def _scale(values: list[float], lower: float | None = None, upper: float | None = None) -> tuple[float, float]:
    finite = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not finite:
        return 0.0, 1.0
    lo = min(finite) if lower is None else lower
    hi = max(finite) if upper is None else upper
    if lo == hi:
        pad = abs(lo) * 0.1 if lo else 1.0
        lo -= pad
        hi += pad
    pad = (hi - lo) * 0.08
    if lower is None:
        lo -= pad
    if upper is None:
        hi += pad
    return lo, hi


def svg_line_chart(
    path: Path,
    title: str,
    x_values: list[float],
    series: list[dict[str, object]],
    y_label: str,
    x_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
    zero_line: bool = False,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for item in series:
        pairs = [
            (x, y)
            for x, y in zip(x_values, item["y"])
            if x is not None and y is not None and not math.isnan(float(y))
        ]
        if pairs:
            cleaned.append({**item, "pairs": pairs})
    if not cleaned:
        if path.exists():
            path.unlink()
        return False

    all_x = [x for item in cleaned for x, _ in item["pairs"]]
    all_y = [y for item in cleaned for _, y in item["pairs"]]
    width, height = 920, 520
    left, right, top, bottom = 82, 36, 52, 84
    plot_w = width - left - right
    plot_h = height - top - bottom
    lo, hi = _scale(all_y, y_min, y_max)
    x_lo, x_hi = min(all_x), max(all_x)

    def sx(x: float) -> float:
        return left + (float(x) - x_lo) / (x_hi - x_lo) * plot_w if x_hi != x_lo else left + plot_w / 2

    def sy(y: float) -> float:
        return top + (hi - float(y)) / (hi - lo) * plot_h

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="600">{title}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
    ]
    for i in range(5):
        y = top + i * plot_h / 4
        val = hi - i * (hi - lo) / 4
        elems.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        elems.append(f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>')
    if zero_line and lo < 0 < hi:
        zy = sy(0)
        elems.append(f'<line x1="{left}" y1="{zy:.2f}" x2="{left + plot_w}" y2="{zy:.2f}" stroke="#777" stroke-width="1.2" stroke-dasharray="5 5"/>')
    for x in sorted(set(all_x)):
        if len(set(all_x)) <= 12 or x in (x_lo, x_hi):
            px = sx(x)
            elems.append(f'<line x1="{px:.2f}" y1="{top + plot_h}" x2="{px:.2f}" y2="{top + plot_h + 5}" stroke="#222"/>')
            elems.append(f'<text x="{px:.2f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{x:g}</text>')
    for idx, item in enumerate(cleaned):
        color = str(item["color"])
        dash = f' stroke-dasharray="{item.get("dash")}"' if item.get("dash") else ""
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in item["pairs"])
        elems.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4"{dash} points="{points}"/>')
        for x, y in item["pairs"]:
            elems.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{color}" stroke="white" stroke-width="1"/>')
        ly = top + 10 + idx * 22
        elems.append(f'<line x1="{left + 10}" y1="{ly}" x2="{left + 34}" y2="{ly}" stroke="{color}" stroke-width="2.4"{dash}/>')
        elems.append(f'<text x="{left + 42}" y="{ly + 4}" font-family="Arial" font-size="13">{item["label"]}</text>')
    elems.append(f'<text x="{left + plot_w/2}" y="{height - 30}" text-anchor="middle" font-family="Arial" font-size="14">{x_label}</text>')
    elems.append(f'<text transform="translate(22 {top + plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial" font-size="14">{y_label}</text>')
    elems.append("</svg>")
    path.write_text("\n".join(elems), encoding="utf-8")
    return True


def save_figures(
    out_dir: Path,
    wide_rows: list[dict[str, object]],
    train_loss: list[dict[str, object]],
    mode: str,
) -> list[str]:
    figures_dir = out_dir / "figures"
    figures: list[str] = []
    x_values = [as_float(row.get("axis_value")) for row in wide_rows]
    if wide_rows and any(x is not None for x in x_values):
        x_label = "Loop epoch" if wide_rows[0].get("axis") == "loop_epoch" else "Test Ratio"
        if svg_line_chart(
            figures_dir / "figure-01-paper-style-relative-delta.svg",
            "Paper-style Relative Delta",
            x_values,
            [
                {"label": "Target MAP@5 Relative Delta", "y": [as_float(row.get("MAP5_target_relative_delta")) for row in wide_rows], "color": OKABE_ITO["blue"]},
                {"label": "Target nDCG@5 Relative Delta", "y": [as_float(row.get("nDCG5_target_relative_delta")) for row in wide_rows], "color": OKABE_ITO["orange"]},
            ],
            "Relative Delta (%)",
            x_label,
            zero_line=True,
        ):
            figures.append("figures/figure-01-paper-style-relative-delta.svg")
        if mode == "loop" and any(as_float(row.get("llm_ratio")) is not None for row in wide_rows):
            if svg_line_chart(
                figures_dir / "figure-03-ratios.svg",
                "llm_ratio and source-composition proxy",
                x_values,
                [
                    {"label": "llm_ratio", "y": [as_float(row.get("llm_ratio")) for row in wide_rows], "color": OKABE_ITO["red"]},
                {"label": "test_ratio / last_ratio proxy", "y": [as_float(row.get("test_ratio")) for row in wide_rows], "color": OKABE_ITO["sky"], "dash": "5 5"},
                {"label": "new_ratio", "y": [as_float(row.get("new_ratio")) for row in wide_rows], "color": OKABE_ITO["purple"], "dash": "2 4"},
                ],
                "Ratio",
                x_label,
                y_min=0,
                y_max=1,
            ):
                figures.append("figures/figure-03-ratios.svg")
    if train_loss and mode in {"train", "unknown"}:
        loss_x = []
        loss_y = []
        for index, row in enumerate(train_loss):
            loop_epoch = as_float(row.get("loop_epoch"))
            epoch = as_float(row.get("epoch"))
            loss_x.append(loop_epoch if loop_epoch is not None else epoch if epoch is not None else float(index))
            loss_y.append(as_float(row.get("loss")))
        if svg_line_chart(
            figures_dir / "figure-04-training-loss.svg",
            "Training loss",
            loss_x,
            [{"label": "Training loss", "y": loss_y, "color": OKABE_ITO["blue"]}],
            "Loss",
            "Loop epoch / train epoch",
        ):
            figures.append("figures/figure-04-training-loss.svg")
    return figures


def write_markdown_summary(
    out_dir: Path,
    run_dir: Path,
    summary: dict[str, object],
    wide_rows: list[dict[str, object]],
    train_loss: list[dict[str, object]],
    figures: list[str],
) -> None:
    lines = [
        "# Rec_SourceBias Analysis Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Run id: `{summary.get('run_id', run_dir.name)}`",
        f"- Return code: `{summary.get('return_code', '')}`",
        f"- Metric points: `{len(wide_rows)}`",
        f"- Train-loss rows: `{len(train_loss)}`",
        "",
        "## Metric Semantics",
        "",
        "- Paper Relative Delta uses `Human Target` / `LLM Target` only.",
        "- Formula: `(Human Target - LLM Target) / mean(Human Target, LLM Target) * 100`.",
        "- Negative values mean `LLM Target` is higher.",
        "- Direct `Human` / `LLM` deltas are diagnostic only.",
        "",
        "## Figures",
        "",
    ]
    if figures:
        lines.extend(f"- `{figure}`" for figure in figures)
    else:
        lines.append("- No metric figures were generated because the run did not emit target metrics.")
    lines.append("")
    (out_dir / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_run(run_dir: Path, out_dir: Path | None = None) -> dict[str, object]:
    out_dir = out_dir or run_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(run_dir / "metrics.csv")
    events = read_csv(run_dir / "events.csv")
    summary = load_summary(run_dir)
    mode = summary_mode(summary)
    tables = derive_metric_tables(metrics, events)
    wide_rows = tables["wide"]
    long_rows = tables["long"]
    train_loss = tables["train_loss"]

    wide_fields = ["axis", "axis_value", "loop_epoch", "test_ratio", "llm_ratio", "new_ratio"]
    for metric in METRIC_COLS:
        wide_fields.extend(
            [
                f"{metric}_human",
                f"{metric}_llm",
                f"{metric}_human_target",
                f"{metric}_llm_target",
                f"{metric}_target_relative_delta",
                f"{metric}_target_relative_delta_pct",
                f"{metric}_target_relative_delta_decimal",
                f"{metric}_target_llm_minus_human",
                f"{metric}_diagnostic_direct_relative_delta",
                f"{metric}_diagnostic_direct_llm_minus_human",
            ]
        )
    generated_tables: list[str] = []
    if write_csv_if_rows(out_dir / "paper_style_metrics_wide.csv", wide_rows, wide_fields):
        generated_tables.append("paper_style_metrics_wide.csv")
    if write_csv_if_rows(
        out_dir / "paper_style_metrics_long.csv",
        long_rows,
        [
            "axis",
            "axis_value",
            "loop_epoch",
            "test_ratio",
            "llm_ratio",
            "new_ratio",
            "metric",
            "target_human",
            "target_llm",
            "target_relative_delta_pct",
            "target_relative_delta_decimal",
            "target_llm_minus_human",
            "direct_human",
            "direct_llm",
            "diagnostic_direct_relative_delta_pct",
            "diagnostic_direct_llm_minus_human",
        ],
    ):
        generated_tables.append("paper_style_metrics_long.csv")
    if write_csv_if_rows(out_dir / "train_loss.csv", train_loss, ["loop_epoch", "epoch", "loss"]):
        generated_tables.append("train_loss.csv")
    summary_rows = summarize(wide_rows, summary)
    if write_csv_if_rows(
        out_dir / "paper_style_summary.csv",
        summary_rows,
        [
            "run_id",
            "metric",
            "points",
            "start_relative_delta",
            "end_relative_delta",
            "min_relative_delta",
            "max_relative_delta",
            "mean_relative_delta",
            "start_llm_minus_human",
            "end_llm_minus_human",
            "min_llm_minus_human",
            "max_llm_minus_human",
        ],
    ):
        generated_tables.append("paper_style_summary.csv")
    stale_run_summary = out_dir / "run_summary.json"
    if stale_run_summary.exists():
        stale_run_summary.unlink()
    stale_gap_figure = out_dir / "figures" / "figure-02-target-gap-trajectory.svg"
    if stale_gap_figure.exists():
        stale_gap_figure.unlink()
    figures = save_figures(out_dir, wide_rows, train_loss, mode)
    write_markdown_summary(out_dir, run_dir, summary, wide_rows, train_loss, figures)
    return {
        "out_dir": str(out_dir),
        "metric_points": len(wide_rows),
        "long_rows": len(long_rows),
        "train_loss_rows": len(train_loss),
        "mode": mode,
        "tables": generated_tables,
        "figures": figures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute paper-style Relative Delta and figures from a Rec_SourceBias saved run."
    )
    parser.add_argument("run_dir", nargs="?", default=None, help="Path to results/source_bias_runs/<run_id>.")
    parser.add_argument("--run-dir", dest="run_dir_opt", default=None, help="Same as positional run_dir.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to <run_dir>/analysis.")
    args = parser.parse_args()

    run_dir_arg = args.run_dir_opt or args.run_dir
    if not run_dir_arg:
        raise SystemExit("run_dir is required")
    run_dir = Path(run_dir_arg).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    result = analyze_run(run_dir, out_dir)
    print(f"Saved analysis: {result['out_dir']}")
    print(f"Metric points: {result['metric_points']}")
    print(f"Train-loss rows: {result['train_loss_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
