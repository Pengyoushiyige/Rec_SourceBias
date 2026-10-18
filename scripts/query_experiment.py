"""Bounded JSON queries: summaries by default, candidate evidence only on request."""
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Iterator, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from candidate_archive import find_candidates, iter_candidate_rows


def jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='One run directory; never modified')
    parser.add_argument('--view', choices=('summaries', 'metrics', 'candidates'), default='summaries')
    parser.add_argument('--round', type=int, dest='round_number')
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--user-id')
    parser.add_argument('--candidate-id')
    parser.add_argument('--limit', type=int, default=20, help='Maximum returned rows (1-1000)')
    args = parser.parse_args()
    if not 1 <= args.limit <= 1000:
        parser.error('--limit must be between 1 and 1000')
    if args.view != 'candidates' and (args.user_id is not None or args.candidate_id is not None):
        parser.error('User/item filters require --view candidates; summaries do not contain event identities')
    if args.view == 'metrics' and args.epoch is not None:
        parser.error('metrics are per feedback round, not per training epoch')
    path = find_candidates(args.run) if args.view == 'candidates' else args.run / (args.view + '.jsonl')
    try:
        source = iter_candidate_rows(path) if args.view == 'candidates' else jsonl_rows(path)
        def matches(row: dict[str, Any]) -> bool:
            return ((args.round_number is None or int(row.get('round', row.get('loop_epoch', -1))) == args.round_number)
                    and (args.epoch is None or int(row.get('training_epoch', -1)) == args.epoch)
                    and (args.user_id is None or row.get('user_id') == args.user_id)
                    and (args.candidate_id is None or row.get('candidate_id') == args.candidate_id))
        try:
            rows = list(islice((row for row in source if matches(row)), args.limit + 1))
        finally:
            source.close()
        print(json.dumps({'source': str(path.resolve()), 'view': args.view,
                          'limit': args.limit, 'truncated': len(rows) > args.limit,
                          'rows': rows[:args.limit]}, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Cannot query experiment: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
