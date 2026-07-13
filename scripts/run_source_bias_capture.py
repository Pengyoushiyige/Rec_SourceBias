import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


METRIC_RE = re.compile(r"^(MAP[135]|nDCG[135]):\s*([-+0-9.eE]+)")
LOSS_RE = re.compile(r"^EPOCH\[(\d+)\]\s+LOSS:([-+0-9.eE]+)")
LLM_RATIO_RE = re.compile(r"^llm ratio:\s*([-+0-9.eE]+)")
NEW_RATIO_RE = re.compile(r"^new ratio:\s*([-+0-9.eE]+)")


def strip_log_prefix(line: str) -> str:
    line = line.rstrip("\n")
    match = re.match(r"^\[[A-Z]+\s+[^\]]+\]\s*(.*)$", line)
    return match.group(1) if match else line


def read_log_text(log_path: Path) -> str:
    data = log_path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16le", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_log(log_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    current: dict[str, object] = {}
    loop_epoch = None

    for raw_line in read_log_text(log_path).splitlines():
        line = strip_log_prefix(raw_line).strip()
        if not line:
            continue

        if line.startswith("Epoch:"):
            try:
                loop_epoch = int(line.split(":", 1)[1].strip())
            except ValueError:
                loop_epoch = line.split(":", 1)[1].strip()
            events.append({"event": "loop_epoch", "loop_epoch": loop_epoch})
            continue

        loss_match = LOSS_RE.match(line)
        if loss_match:
            events.append(
                {
                    "event": "train_loss",
                    "loop_epoch": loop_epoch,
                    "epoch": int(loss_match.group(1)),
                    "loss": float(loss_match.group(2)),
                }
            )
            continue

        llm_ratio_match = LLM_RATIO_RE.match(line)
        if llm_ratio_match:
            events.append(
                {
                    "event": "llm_ratio",
                    "loop_epoch": loop_epoch,
                    "llm_ratio": float(llm_ratio_match.group(1)),
                }
            )
            continue

        new_ratio_match = NEW_RATIO_RE.match(line)
        if new_ratio_match:
            events.append(
                {
                    "event": "new_ratio",
                    "loop_epoch": loop_epoch,
                    "new_ratio": float(new_ratio_match.group(1)),
                }
            )
            continue

        if line.startswith("Test Ratio:"):
            current = {
                "loop_epoch": loop_epoch,
                "test_ratio": float(line.split(":", 1)[1].strip()),
            }
            continue

        if line.startswith("Test Type:"):
            current["test_type"] = line.split(":", 1)[1].strip()
            continue

        metric_match = METRIC_RE.match(line)
        if metric_match and current:
            current[metric_match.group(1)] = float(metric_match.group(2))
            if all(key in current for key in ("MAP1", "MAP3", "MAP5", "nDCG1", "nDCG3", "nDCG5")):
                metrics.append(dict(current))
                current = {}

    return metrics, events


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_mode(command: list[str], fallback: str) -> str:
    """Read --mode from an explicit wrapped command when present."""
    try:
        value = command[command.index("--mode") + 1]
    except (ValueError, IndexError):
        return fallback
    return value if value in {"train", "test", "loop"} else fallback


def default_command(args: argparse.Namespace, ckpt_dir: str) -> list[str]:
    python_exe = args.python or sys.executable
    command = [
        python_exe,
        "src/run.py",
        "--mode",
        args.mode,
        "--dataset",
        args.dataset,
        "--behaviors_file",
        args.behaviors_file,
        "--behaviors_file1",
        args.behaviors_file1,
        "--behaviors_file2",
        args.behaviors_file2,
        "--test_behaviors_file",
        args.test_behaviors_file,
        "--human_news_file",
        args.human_news_file,
        "--llm_news_file",
        args.llm_news_file,
        "--loop_epochs",
        str(args.loop_epochs),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--news_batch_size",
        str(args.news_batch_size),
        "--gpu",
        args.gpu,
        "--num_workers",
        str(args.num_workers),
        "--model_type",
        args.model_type,
        "--load_ckpt_name",
        args.load_ckpt_name,
        "--seed",
        str(args.seed),
    ]
    if ckpt_dir:
        command.extend(["--ckpt_dir", ckpt_dir])
    if args.debias:
        command.extend(
            [
                "--debias",
                "--debias_type",
                args.debias_type,
                "--llm_rewirte_news_file",
                args.llm_rewirte_news_file,
            ]
        )
    return command


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        description="Run a Rec_SourceBias scenario and save raw logs plus parsed metrics."
    )
    parser.add_argument("--run-name", default="source_bias_loop")
    parser.add_argument("--output-root", default="results/source_bias_runs")
    parser.add_argument("--python", default=None, help="Python executable. Defaults to the current interpreter.")
    parser.add_argument("--mode", default="loop", choices=["loop", "train", "test"], help="Running mode.")
    parser.add_argument("--load_ckpt_name", default="epoch-5.pt")
    parser.add_argument("--ckpt_dir", default="")
    parser.add_argument("--behaviors_file", default="behaviors.tsv")
    parser.add_argument("--model_type", default="BERT4Rec")
    parser.add_argument("--dataset", default="Amazon_Beauty_sample")
    parser.add_argument("--behaviors_file1", default="behaviors1.tsv")
    parser.add_argument("--behaviors_file2", default="behaviors2.tsv")
    parser.add_argument("--test_behaviors_file", default="behaviors.tsv")
    parser.add_argument("--human_news_file", default="news_parsed_bert-base-uncased.tsv")
    parser.add_argument("--llm_news_file", default="news_llama_parsed_bert-base-uncased.tsv")
    parser.add_argument("--llm_rewirte_news_file", default="news_parsed_bert-base-uncased.tsv")
    parser.add_argument("--loop_epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--news_batch_size", type=int, default=16)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--dry-run", action="store_true", help="Print and save the command without executing it.")
    parser.add_argument("--no-analysis", action="store_true", help="Skip automatic paper-style analysis after capture.")
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--debias_type", default="emb_entropy")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional explicit command after --.")
    args = parser.parse_args()

    explicit_command = args.command[1:] if args.command[:1] == ["--"] else args.command
    requested_mode = command_mode(explicit_command, args.mode) if explicit_command else args.mode
    run_name = args.run_name
    if run_name == "source_bias_loop" and requested_mode != "loop":
        run_name = f"source_bias_{requested_mode}"

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + run_name
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = explicit_command
    default_command_used = not command
    if not command:
        ckpt_dir = args.ckpt_dir
        if not ckpt_dir and args.mode == "train":
            ckpt_dir = str(output_dir)
        command = default_command(args, ckpt_dir)
    run_mode = command_mode(command, args.mode)

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    if default_command_used:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
    started_at = datetime.now().isoformat(timespec="seconds")
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(f"# started_at: {started_at}\n")
        log_file.write(f"# cwd: {Path.cwd()}\n")
        log_file.write(f"# command: {' '.join(command)}\n\n")
        log_file.flush()

        if args.dry_run:
            log_file.write("# dry_run: true\n")
            return_code = 0
        else:
            process = subprocess.Popen(
                command,
                cwd=Path.cwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            return_code = process.wait()

    metrics, events = parse_log(log_path)
    artifacts = ["run.log", "summary.json"]
    if metrics:
        write_csv(
            output_dir / "metrics.csv",
            metrics,
            ["loop_epoch", "test_ratio", "test_type", "MAP1", "MAP3", "MAP5", "nDCG1", "nDCG3", "nDCG5"],
        )
        with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
            for row in metrics:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        artifacts.extend(["metrics.csv", "metrics.jsonl"])
    if events:
        write_csv(output_dir / "events.csv", events, ["event", "loop_epoch", "epoch", "loss", "llm_ratio", "new_ratio"])
        artifacts.append("events.csv")
    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "return_code": return_code,
        "mode": run_mode,
        "command": command,
        "log_path": str(log_path),
        "metrics_rows": len(metrics),
        "events_rows": len(events),
        "dry_run": args.dry_run,
        "artifacts": artifacts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_analysis and not args.dry_run and (metrics or events):
        try:
            from compute_paper_style_metrics import analyze_run

            analysis = analyze_run(output_dir)
            summary["analysis"] = analysis
            summary["artifacts"].extend(
                ["analysis/analysis_summary.md"]
                + [f"analysis/{name}" for name in analysis["tables"]]
                + [f"analysis/{name}" for name in analysis["figures"]]
            )
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Saved analysis: {analysis['out_dir']}")
        except Exception as exc:
            summary["analysis_error"] = repr(exc)
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Analysis failed: {exc}", file=sys.stderr)
    print(f"\nSaved raw log: {log_path}")
    if metrics:
        print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    if events:
        print(f"Saved events: {output_dir / 'events.csv'}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
