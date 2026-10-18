"""Versioned, normalized click evidence and a streaming legacy CSV adapter.

Catalogs grow with unique users/items, not event count. Events retain candidate
positions (including repeated IDs), binary64 values and actual sampling counts.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

ARCHIVE_NAME = 'candidate_events.jsonl.gz'
SCHEMA = 'click-candidate-events-v1'
FIELDS = ('round', 'training_epoch', 'event_key', 'user_id', 'candidate_id',
          'source', 'score', 'label', 'rank', 'cue', 'alpha', 'probability',
          'clicked', 'training_count', 'in_top3', 'in_top5')


def find_candidates(directory: Path) -> Path:
    """Prefer new evidence; never silently fall back if that archive is broken."""
    compact = directory / ARCHIVE_NAME
    return compact if compact.exists() else directory / 'candidates.csv'


class CandidateArchive:
    def __init__(self, directory: Path, click_model: str) -> None:
        self._stream = gzip.open(directory / ARCHIVE_NAME, 'xt', encoding='utf-8', compresslevel=6)
        self._users: dict[tuple[str, float | None], int] = {}
        self._candidates: dict[tuple[str, float | None], int] = {}
        self._write({'schema': SCHEMA, 'click_model': click_model})

    def _write(self, record: dict[str, Any]) -> None:
        self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False,
                                      separators=(',', ':')) + '\n')

    def record(self, *, round_number: int, training_epoch: int, event_key: Any,
               user_id: Any, candidate_ids: Sequence[str], selected: int,
               training_counts: Sequence[int], diagnostics: dict[str, Any] | None = None) -> None:
        user = (str(user_id), float(diagnostics['alpha']) if diagnostics else None)
        if user not in self._users:
            self._users[user] = len(self._users)
            self._write({'type': 'user', 'user_id': user[0], 'alpha': user[1]})
        indices = []
        for i, candidate_id in enumerate(candidate_ids):
            candidate = (candidate_id, float(diagnostics['cues'][i]) if diagnostics else None)
            if candidate not in self._candidates:
                self._candidates[candidate] = len(self._candidates)
                self._write({'type': 'candidate', 'candidate_id': candidate[0], 'cue': candidate[1]})
            indices.append(self._candidates[candidate])
        record = {'type': 'event', 'round': round_number, 'training_epoch': training_epoch,
                  'event_key': event_key, 'user': self._users[user], 'candidates': indices,
                  'selected': selected,
                  'training': [[i, int(n)] for i, n in enumerate(training_counts) if n]}
        if diagnostics is not None:
            for stored, original, cast in (('scores', 'scores', float), ('labels', 'labels', int),
                                           ('ranks', 'ranks', int), ('probabilities', 'probabilities', float)):
                record[stored] = [cast(value) for value in diagnostics[original]]
        self._write(record)

    def flush(self) -> None:
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()


def _index(value: Any, size: int) -> int:
    if type(value) is not int or not 0 <= value < size:
        raise ValueError('Invalid candidate archive index')
    return value


def _compact_rows(path: Path) -> Iterator[dict[str, str]]:
    users, candidates = [], []
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream, '{}'))
        if not isinstance(header, dict) or header.get('schema') != SCHEMA:
            raise ValueError('Unsupported candidate archive schema')
        for line in stream:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError('Invalid candidate archive record')
            kind = record['type']
            if kind == 'user':
                users.append((record['user_id'], record['alpha']))
                continue
            if kind == 'candidate':
                candidate_id = record['candidate_id']
                if not isinstance(candidate_id, str) or candidate_id.rsplit('-', 1)[-1] not in ('human', 'llm'):
                    raise ValueError('Invalid candidate archive source suffix')
                candidates.append((candidate_id, record['cue']))
                continue
            if kind != 'event':
                raise ValueError('Unknown candidate archive record type')
            if (any(type(record[name]) is not int or record[name] < 1 for name in ('round', 'training_epoch'))
                    or not isinstance(record['event_key'], str)):
                raise ValueError('Invalid candidate archive event identity')
            user_id, alpha = users[_index(record['user'], len(users))]
            indices = record['candidates']
            if not isinstance(indices, list) or not indices:
                raise ValueError('Empty candidate archive event')
            size = len(indices)
            selected = _index(record['selected'], size)
            counts = [0] * size
            seen = set()
            for position, count in record['training']:
                position = _index(position, size)
                if position in seen or type(count) is not int or count <= 0:
                    raise ValueError('Invalid candidate training multiplicity')
                seen.add(position)
                counts[position] = count
            scored = 'scores' in record
            arrays = ('scores', 'labels', 'ranks', 'probabilities')
            if (scored and any(not isinstance(record.get(name), list) or len(record[name]) != size
                               for name in arrays)) or (not scored and any(name in record for name in arrays)):
                raise ValueError('Misaligned candidate archive arrays')
            for i, index in enumerate(indices):
                candidate_id, cue = candidates[_index(index, len(candidates))]
                if scored and (alpha is None or cue is None):
                    raise ValueError('Scored candidate is missing alpha/cue')
                row = {field: '' for field in FIELDS}
                row.update(round=str(record['round']), training_epoch=str(record['training_epoch']),
                           event_key=record['event_key'],
                           user_id=str(user_id), candidate_id=candidate_id,
                           source=candidate_id.rsplit('-', 1)[1], clicked=str(int(i == selected)),
                           training_count=str(counts[i]))
                if scored:
                    row.update(score=str(record['scores'][i]), label=str(record['labels'][i]),
                               rank=str(record['ranks'][i]), probability=str(record['probabilities'][i]),
                               alpha=str(alpha), cue=str(cue),
                               in_top3=str(int(record['ranks'][i] <= 3)),
                               in_top5=str(int(record['ranks'][i] <= 5)))
                yield row


def iter_candidate_rows(path: Path, required: Sequence[str] = FIELDS) -> Iterator[dict[str, str]]:
    """Yield legacy-compatible rows without materializing a complete archive."""
    path = Path(path)
    if path.is_dir():
        path = find_candidates(path)
    if path.name == ARCHIVE_NAME:
        try:
            yield from _compact_rows(path)
        except (KeyError, TypeError, IndexError, EOFError) as error:
            raise ValueError(f'Invalid candidate archive: {error}') from error
    else:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not set(required).issubset(reader.fieldnames or []):
                raise ValueError('missing candidate diagnostics columns')
            yield from reader
