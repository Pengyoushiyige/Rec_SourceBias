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


def strip_log_prefix(line: str) -> str:
    line = line.rstrip("\n")
    match = re.match(r"^\[[A-Z]+\s+[^\]]+\]\s*(.*)$", line)
    return match.group(1) if match else line


def parse_log(log_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    metrics: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    current: dict[str, object] = {}
    loop_epoch = None

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def default_command(args: argparse.Namespace) -> list[str]:
    python_exe = args.python or sys.executable
    command = [
        python_exe,
        "src/run.py",
        "--mode",
        "loop",
        "--dataset",
        args.dataset,
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
        "--gpu",
        args.gpu,
        "--num_workers",
        str(args.num_workers),
    ]
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
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--debias", action="store_true")
    parser.add_argument("--debias_type", default="emb_entropy")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Optional explicit command after --.")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + args.run_name
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    default_command_used = not command
    if not command:
        command = default_command(args)

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
    write_csv(
        output_dir / "metrics.csv",
        metrics,
        ["loop_epoch", "test_ratio", "test_type", "MAP1", "MAP3", "MAP5", "nDCG1", "nDCG3", "nDCG5"],
    )
    write_csv(output_dir / "events.csv", events, ["event", "loop_epoch", "epoch", "loss", "llm_ratio"])
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "return_code": return_code,
        "command": command,
        "log_path": str(log_path),
        "metrics_rows": len(metrics),
        "events_rows": len(events),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved raw log: {log_path}")
    print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    print(f"Saved events: {output_dir / 'events.csv'}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
