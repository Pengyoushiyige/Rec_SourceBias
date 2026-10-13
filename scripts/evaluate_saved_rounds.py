"""Evaluate saved user encoders with stateless common history assignments (no training/download)."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from round_observations import SCHEMA, sha256
from metrics import fast_ndcg_score, fast_map_score, fast_recall_score

NAMESPACE = 'sacm-controlled-evaluation-v1'


def uniform(seed, *key):
    raw = json.dumps([NAMESPACE, int(seed), *key], ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big') / 2**64


def history_sources(seed, history, length, ratio):
    ids = history.split()[-length:]
    return [uniform(seed, 'history', history, i) < ratio for i in range(len(ids))]


def read_behaviors(path):
    rows = []
    with open(path, encoding='utf-8') as stream:
        for number, line in enumerate(stream):
            fields = line.rstrip('\r\n').split('\t')
            if len(fields) != 5:
                raise ValueError('Expected 5 TSV fields at row %s' % number)
            history = fields[3] if fields[3] else ' '
            candidates, labels = [], []
            for token in fields[4].split():
                item, label = token.rsplit('-', 1)
                candidates.append(item)
                labels.append(int(label))
            rows.append((number, fields[0], history, candidates, np.asarray(labels)))
    return rows


@torch.no_grad()
def evaluate(encoder, args, vectors, rows, ratio, seed, device):
    users = {}
    histories = list(dict.fromkeys(row[2] for row in rows))
    assignment_digest = hashlib.sha256()
    for offset in range(0, len(histories), args.batch_size):
        batch = histories[offset:offset + args.batch_size]
        logs, masks = [], []
        for history in batch:
            ids = history.split()[-args.user_log_length:]
            sources = history_sources(seed, history, args.user_log_length, ratio)
            assignment_digest.update(json.dumps([history, sources], ensure_ascii=False).encode())
            values = [vectors['llm' if source else 'human'][item] for item, source in zip(ids, sources)]
            padded = args.user_log_length - len(ids)
            values += [vectors['human']['PADDED_NEWS']] * padded
            logs.append(torch.stack(values))
            masks.append([1] * len(ids) + [0] * padded)
        encoded = encoder(torch.stack(logs).to(device), torch.tensor(masks, device=device)).cpu()
        users.update(zip(batch, encoded))
    totals = {name: np.zeros(8) for name in ['Human', 'LLM', 'Human Target', 'LLM Target']}
    counts = {'total': len(rows), 'all_zero': 0, 'all_one': 0, 'first_score_equal': 0, 'evaluated': 0}
    human_wins = llm_wins = 0
    exposures = {3: [], 5: []}
    for number, impression, history, candidates, labels in rows:
        if not len(labels) or np.all(labels == 0):
            counts['all_zero'] += 1
            continue
        if np.all(labels == 1):
            counts['all_one'] += 1
            continue
        user = users[history].numpy()
        human = np.dot(torch.stack([vectors['human'][item] for item in candidates]).numpy(), user)
        llm = np.dot(torch.stack([vectors['llm'][item] for item in candidates]).numpy(), user)
        if human[0] == llm[0]:
            counts['first_score_equal'] += 1
            continue
        human_wins += int(np.sum((human - llm)[labels == 1] > 0))
        llm_wins += int(np.sum((human - llm)[labels == 1] < 0))
        llm_first = uniform(seed, 'candidate_mix', number, impression, history) < .5
        mix = np.concatenate((llm, human) if llm_first else (human, llm))
        human_labels = np.concatenate((np.zeros_like(labels), labels) if llm_first else (labels, np.zeros_like(labels)))
        llm_labels = np.concatenate((labels, np.zeros_like(labels)) if llm_first else (np.zeros_like(labels), labels))
        sources = np.concatenate((np.ones_like(labels), np.zeros_like(labels)) if llm_first else (np.zeros_like(labels), np.ones_like(labels)))
        order = np.argsort(mix)[::-1]  # Preserve native metric ordering, including duplicated positions.
        for k in exposures:
            exposures[k].append(float(np.mean(sources[order[:k]])))
        for name, truth, scores in [('Human', labels, human), ('LLM', labels, llm),
                                    ('Human Target', human_labels, mix), ('LLM Target', llm_labels, mix)]:
            totals[name] += np.array(fast_ndcg_score(truth, scores, [1, 3, 5]) +
                                     fast_map_score(truth, scores, [1, 3, 5]) +
                                     fast_recall_score(truth, scores, (3, 5)))
        counts['evaluated'] += 1
    names = ['nDCG1', 'nDCG3', 'nDCG5', 'MAP1', 'MAP3', 'MAP5', 'Recall3', 'Recall5']
    metrics = {name: dict(zip(names, (value / counts['evaluated']).tolist()))
               for name, value in totals.items()} if counts['evaluated'] else {}
    return {'metrics': metrics, 'counts': counts, 'history_assignment_sha256': assignment_digest.hexdigest(),
            'llm_pair_win_ratio': llm_wins / (human_wins + llm_wins) if human_wins + llm_wins else None,
            'exposure': {str(k): float(np.mean(v)) if v else None for k, v in exposures.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratios', default='0.5', help='Comma-separated native,0.5,0,1 or other [0,1] values')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--assignment-seed', type=int, help='Defaults to saved experimental seed; share across paired conditions')
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError(options.output)
    manifest = json.loads((options.artifacts / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported artifact schema')
    vector_path = options.artifacts / manifest['vectors_file']
    behavior_path = Path(manifest['test_behaviors_file'])
    for path, expected in [(vector_path, manifest['vectors_sha256']), (behavior_path, manifest['test_behaviors_sha256'])]:
        if sha256(path) != expected:
            raise ValueError('Hash mismatch: %s' % path)
    vectors = torch.load(vector_path, map_location='cpu', weights_only=True)
    rows = read_behaviors(behavior_path)
    snapshots = sorted(options.artifacts.glob('round-*.pt'))
    if not snapshots:
        raise ValueError('No saved rounds; legacy pilot cannot be re-evaluated')
    ratios = options.ratios.split(',')
    for ratio in ratios:
        if ratio != 'native' and not 0 <= float(ratio) <= 1:
            raise ValueError('Ratio outside [0,1]')
    options.output.mkdir(parents=True, exist_ok=False)
    import utils  # Model construction only; no PLM encoding/loading.
    results = []
    for snapshot in snapshots:
        saved = torch.load(snapshot, map_location='cpu', weights_only=True)
        if saved['schema'] != SCHEMA or saved['vectors_sha256'] != manifest['vectors_sha256'] or saved['test_behaviors_sha256'] != manifest['test_behaviors_sha256']:
            raise ValueError('Inconsistent snapshot: %s' % snapshot)
        args = SimpleNamespace(**saved['args'])
        encoder = utils.get_model(args.model_type)(args, None).user_encoder
        encoder.load_state_dict(saved['user_encoder_state_dict'], strict=True)
        encoder.to(options.device).eval()
        seed = options.assignment_seed if options.assignment_seed is not None else args.seed
        for ratio_label in ratios:
            ratio = saved['native_feedback_ratio'] if ratio_label == 'native' else float(ratio_label)
            result = evaluate(encoder, args, vectors, rows, ratio, seed, options.device)
            result.update(round=saved['round'], ratio_label=ratio_label, history_ratio=ratio,
                          native_feedback_ratio=saved['native_feedback_ratio'], assignment_seed=seed,
                          checkpoint_sha256=sha256(snapshot))
            results.append(result)
    with open(options.output / 'metrics.csv', 'x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['round', 'ratio_label', 'history_ratio', 'metric_group', 'metric', 'value'])
        writer.writeheader()
        for result in results:
            for group, metrics in result['metrics'].items():
                for metric, value in metrics.items():
                    writer.writerow({k: result[k] for k in ['round', 'ratio_label', 'history_ratio']} | {'metric_group': group, 'metric': metric, 'value': value})
    (options.output / 'analysis.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    (options.output / 'manifest.json').write_text(json.dumps({'schema': SCHEMA, 'namespace': NAMESPACE,
        'artifacts': str(options.artifacts.resolve()), 'device': options.device, 'input_manifest': manifest, 'evaluator_sha256': sha256(__file__),
        'interpretation': 'Controlled-evaluator diagnostic; native uses actual ratio with new fixed draws, not exact native log reproduction.'}, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
