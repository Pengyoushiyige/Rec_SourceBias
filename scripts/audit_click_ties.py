"""Read-only exact-score tie audit; candidate positions retain multiplicity."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def new_counts() -> dict[str, int]:
    return dict(scored_impressions=0, initial_unscored_impressions=0,
                paired_positions=0, exact_tied_pairs=0, paired_tie_impressions=0,
                cross_source_tie_impressions=0, tied_pair_human_first=0,
                tied_pair_llm_first=0)


def count_event(rows: list[dict[str, str]]) -> dict[str, int]:
    counts = new_counts()
    present = [row['score'] != '' for row in rows]
    if not any(present):
        counts['initial_unscored_impressions'] = 1
        return counts
    if not all(present):
        raise ValueError('Partially scored impression')
    counts['scored_impressions'] = 1
    groups: dict[tuple[str, int], dict[str, list[tuple[float, int]]]] = defaultdict(
        lambda: {'human': [], 'llm': []})
    score_sources: dict[float, set[str]] = defaultdict(set)
    ranks = []
    for row in rows:
        base, source = row['candidate_id'].rsplit('-', 1)
        if source not in ('human', 'llm') or row['source'] != source:
            raise ValueError('Candidate/source mismatch')
        score, rank, label = float(row['score']), int(row['rank']), int(row['label'])
        if not math.isfinite(score) or label not in (0, 1):
            raise ValueError('Non-finite score or invalid label')
        ranks.append(rank)
        groups[(base, label)][source].append((score, rank))
        score_sources[score].add(source)
    if sorted(ranks) != list(range(1, len(rows) + 1)):
        raise ValueError('Ranks are not a candidate-position permutation')
    ordered_scores = [float(row['score']) for row in sorted(rows, key=lambda r: int(r['rank']))]
    if any(a < b for a, b in zip(ordered_scores, ordered_scores[1:])):
        raise ValueError('Ranks violate descending score order')
    counts['cross_source_tie_impressions'] = int(any(len(s) > 1 for s in score_sources.values()))
    for versions in groups.values():
        human, llm = versions['human'], versions['llm']
        if len(human) != len(llm):
            raise ValueError('Unpaired (base item, label) multiplicity')
        # Duplicate positions of one source/ID must describe the same scored item.
        # Without occurrence IDs we cannot infer a unique cross-source rank pairing
        # when duplicate instances have distinct scores; reject rather than invent it.
        if len({s for s, _ in human}) > 1 or len({s for s, _ in llm}) > 1:
            raise ValueError('Ambiguous duplicate scores; explicit occurrence IDs required')
        counts['paired_positions'] += len(human)
        # Match repeated copies by occurrence in CSV order, not by ID deduplication.
        for (hs, hr), (ls, lr) in zip(human, llm):
            if hs == ls:
                counts['exact_tied_pairs'] += 1
                counts['tied_pair_human_first'] += int(hr < lr)
                counts['tied_pair_llm_first'] += int(lr < hr)
    counts['paired_tie_impressions'] = int(counts['exact_tied_pairs'] > 0)
    return counts


def with_ratios(counts: dict[str, int]) -> dict[str, Any]:
    result: dict[str, Any] = dict(counts)
    for field, num, denom in (
        ('paired_exact_tie_rate', 'exact_tied_pairs', 'paired_positions'),
        ('paired_tie_impression_rate', 'paired_tie_impressions', 'scored_impressions'),
        ('cross_source_tie_impression_rate', 'cross_source_tie_impressions', 'scored_impressions'),
        ('tied_pair_human_first_rate', 'tied_pair_human_first', 'exact_tied_pairs')):
        result[field] = counts[num] / counts[denom] if counts[denom] else None
    return result


def audit_file(path: Path) -> dict[str, Any]:
    rounds: dict[tuple[int, int], dict[str, int]] = {}
    total = new_counts()
    seen: set[tuple[str, str, str]] = set()
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        required = {'round', 'training_epoch', 'event_key', 'candidate_id', 'source', 'score', 'rank', 'label'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing candidate CSV columns')
        for key, iterator in itertools.groupby(reader, lambda r: (r['round'], r['training_epoch'], r['event_key'])):
            if key in seen:
                raise ValueError(f'Non-contiguous repeated event: {key}')
            seen.add(key)
            try:
                rows = list(iterator)
                if int(key[0]) < 1 or int(key[1]) < 1:
                    raise ValueError('Round/training epoch must be positive integers')
                event = json.loads(key[2])
                initial = isinstance(event, list) and bool(event) and event[0] == 'initial'
                if (int(key[0]) == 1) != initial:
                    raise ValueError('Round/initial event identity mismatch')
                if initial and any(row['score'] != '' for row in rows):
                    raise ValueError('Initialization unexpectedly has scores')
                if not initial and not any(row['score'] != '' for row in rows):
                    raise ValueError('Simulated impression has no scores')
                counts = count_event(rows)
            except (ValueError, KeyError) as error:
                raise ValueError(f'{path} impression {key}: {error}') from error
            epoch = (int(key[0]), int(key[1]))
            aggregate = rounds.setdefault(epoch, new_counts())
            for field, value in counts.items():
                aggregate[field] += value
                total[field] += value
    if not seen:
        raise ValueError('Empty candidate CSV')
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
    return dict(path=str(path.resolve()), sha256=sha256(path),
                condition=manifest.get('condition'), seed=manifest.get('seed'),
                manifest_status=manifest.get('status'), totals=with_ratios(total),
                epochs=[dict(round=r, training_epoch=e, **with_ratios(c)) for (r, e), c in sorted(rounds.items())])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='candidates.csv, run directory or cohort directory')
    parser.add_argument('--output', type=Path, help='New JSON file; existing outputs are never overwritten')
    args = parser.parse_args()
    if args.output and args.output.exists():
        parser.error('Output already exists')
    paths = [args.input] if args.input.is_file() else sorted(args.input.rglob('candidates.csv'))
    if not paths:
        parser.error('No candidate CSVs found')
    runs = [audit_file(path) for path in paths]
    total = {key: sum(run['totals'][key] for run in runs) for key in new_counts()}
    payload = dict(schema_version=1, equality='exact finite score equality, no tolerance',
                   pair_unit='(base item, label) per-source CSV occurrence; repeated positions retained',
                   scope='logged click ranking, not native evaluation or unscored initialization',
                   runs=runs, totals=with_ratios(total))
    output = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            handle.write(output + '\n')
    else:
        print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
