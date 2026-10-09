"""Streaming click/training observations; ratios have explicit counted denominators."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class ClickObservations:
    """One run writer, with independent summaries for each reconstructed dataset."""

    FIELDS = ["round", "training_epoch", "event_key", "user_id", "candidate_id",
              "source", "score", "label", "rank", "cue", "alpha", "probability",
              "clicked", "training_count", "in_top3", "in_top5"]

    def __init__(self, directory: str | Path, click_model: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.click_model = click_model
        # Existing data must never be mixed with a new run.
        self._rows = (self.directory / "candidates.csv").open("x", newline="", encoding="utf-8")
        try:
            self._summaries = (self.directory / "summaries.jsonl").open("x", encoding="utf-8")
        except BaseException:
            self._rows.close()
            raise
        self._writer = csv.DictWriter(self._rows, fieldnames=self.FIELDS)
        self._writer.writeheader()
        self._counts: dict[str, Any] = {}

    def start_epoch(self, round_number: int, training_epoch: int) -> None:
        self._counts = {"schema_version": 1, "click_model": self.click_model,
                        "round": round_number, "training_epoch": training_epoch,
                        "impressions": 0, "clicks": 0, "click_llm": 0,
                        "training_candidates": 0, "training_llm": 0,
                        "training_positive_llm": 0, "simulated_impressions": 0,
                        "alpha_sum": 0.0, "cue_human_sum": 0.0, "cue_human_count": 0,
                        "cue_llm_sum": 0.0, "cue_llm_count": 0,
                        "exposure_llm_at3": 0, "exposure_count_at3": 0,
                        "exposure_llm_at5": 0, "exposure_count_at5": 0}

    def _ensure_epoch(self, round_number: int, training_epoch: int) -> None:
        if not self._counts:
            self.start_epoch(round_number, training_epoch)
        if (self._counts["round"], self._counts["training_epoch"]) != (round_number, training_epoch):
            raise ValueError("finish_epoch must precede starting another observation epoch")

    def _training(self, sources: Sequence[str]) -> None:
        self._counts["impressions"] += 1
        self._counts["clicks"] += 1
        self._counts["click_llm"] += int(sources[0] == "llm")
        self._counts["training_positive_llm"] += int(sources[0] == "llm")
        self._counts["training_candidates"] += len(sources)
        self._counts["training_llm"] += sum(source == "llm" for source in sources)

    def record_initial(self, candidate_ids: Sequence[str], user_id: object,
                       impression: object, round_number: int, training_epoch: int) -> None:
        self._ensure_epoch(round_number, training_epoch)
        self._training([value.rsplit("-", 1)[1] for value in candidate_ids])
        for i, candidate_id in enumerate(candidate_ids):
            self._writer.writerow({"round": round_number, "training_epoch": training_epoch,
                "event_key": json.dumps(["initial", str(impression), str(user_id)]),
                "user_id": str(user_id), "candidate_id": candidate_id,
                "source": candidate_id.rsplit("-", 1)[1], "clicked": int(i == 0),
                "training_count": 1})

    def record(self, diagnostics: dict[str, Any], neg_indices: Sequence[int], *,
               round_number: int, training_epoch: int, event_key: object) -> None:
        self._ensure_epoch(round_number, training_epoch)
        d = diagnostics
        selected = int(d["selected_index"])
        sources = list(d["sources"])
        self._training([sources[selected]] + [sources[int(i)] for i in neg_indices])
        self._counts["simulated_impressions"] += 1
        self._counts["alpha_sum"] += float(d["alpha"])
        ranks = np.asarray(d["ranks"])
        for k in (3, 5):
            top_sources = [s for s, rank in zip(sources, ranks) if rank <= k]
            self._counts[f"exposure_count_at{k}"] += len(top_sources)
            self._counts[f"exposure_llm_at{k}"] += sum(s == "llm" for s in top_sources)
        for i, candidate in enumerate(d["candidate_ids"]):
            source = sources[i]
            cue = float(d["cues"][i])
            self._counts[f"cue_{source}_sum"] += cue
            self._counts[f"cue_{source}_count"] += 1
            self._writer.writerow({"round": round_number, "training_epoch": training_epoch,
                "event_key": json.dumps(event_key, ensure_ascii=False, separators=(",", ":")),
                "user_id": d["user_id"], "candidate_id": candidate, "source": source,
                "score": float(d["scores"][i]), "label": int(d["labels"][i]),
                "rank": int(ranks[i]), "cue": cue, "alpha": float(d["alpha"]),
                "probability": float(d["probabilities"][i]), "clicked": int(i == selected),
                "training_count": int(i == selected) + sum(int(j) == i for j in neg_indices),
                "in_top3": int(ranks[i] <= 3), "in_top5": int(ranks[i] <= 5)})

    def finish_epoch(self) -> dict[str, Any]:
        if not self._counts:
            raise ValueError("No observation epoch started")
        c = dict(self._counts)
        def ratio(numerator: str, denominator: str) -> float | None:
            return c[numerator] / c[denominator] if c[denominator] else None
        c["click_ratio"] = ratio("click_llm", "clicks")
        c["training_ratio"] = ratio("training_llm", "training_candidates")
        c["training_positive_ratio"] = ratio("training_positive_llm", "clicks")
        c["alpha_mean"] = ratio("alpha_sum", "simulated_impressions")
        for source in ("human", "llm"):
            c[f"cue_{source}_mean"] = ratio(f"cue_{source}_sum", f"cue_{source}_count")
        for k in (3, 5):
            c[f"exposure_ratio_at{k}"] = ratio(f"exposure_llm_at{k}", f"exposure_count_at{k}")
        self._summaries.write(json.dumps(c, ensure_ascii=False, allow_nan=False) + "\n")
        self._summaries.flush()
        self._rows.flush()
        self._counts = {}
        return c

    def close(self) -> None:
        self._rows.close()
        self._summaries.close()

    def __enter__(self) -> ClickObservations:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
