"""Passive round artifacts: no inference, RNG, optimizer or model mode changes."""
import hashlib
import json
from pathlib import Path

import torch

SCHEMA = 'sacm-round-observations-v1'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_new(path, payload):
    with open(path, 'x', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)


class RoundObservations:
    def __init__(self, directory, args, human_vectors, llm_vectors):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.args = dict(vars(args))
        self.behaviors = Path(args.test_behaviors_file).resolve()
        self.behaviors_sha = sha256(self.behaviors)
        self.provenance = {"torch_version": str(torch.__version__),
            "code_sha256": {name: sha256(Path(__file__).with_name(name))
                for name in ["train_loop.py", "round_observations.py", "Dataset.py", "click_models.py"]},
            "text_inputs": {name: {"path": str(Path(getattr(args, name)).resolve()),
                                    "sha256": sha256(getattr(args, name))}
                for name in ["human_news_file", "llm_news_file"] if getattr(args, name, None)},
            "vector_cache_source": "Trainer.get_news2vector precomputed tensors; no re-encoding",
            "news_batch_size": getattr(args, "news_batch_size", None)}
        vector_path = self.directory / 'text_vectors.pt'
        with open(vector_path, 'xb') as stream:
            torch.save({'human': {k: v.detach().cpu() for k, v in human_vectors.items()},
                        'llm': {k: v.detach().cpu() for k, v in llm_vectors.items()}}, stream)
        self.vector_sha = sha256(vector_path)
        _json_new(self.directory / 'manifest.json', {
            'schema': SCHEMA, 'args': self.args, 'provenance': self.provenance,
            'vectors_file': vector_path.name, 'vectors_sha256': self.vector_sha,
            'test_behaviors_file': str(self.behaviors), 'test_behaviors_sha256': self.behaviors_sha,
            'notes': 'Cached vectors exactly as used in training; encoder mode unchanged; no PLM checkpoint.'})

    def save_round(self, round_number, user_encoder, native_ratio, native_eval_diagnostics=None):
        path = self.directory / ('round-%03d.pt' % round_number)
        payload = {'schema': SCHEMA, 'round': round_number, 'args': self.args, 'provenance': self.provenance,
                   'native_feedback_ratio': float(native_ratio),
                   'native_eval_diagnostics': native_eval_diagnostics,
                   'vectors_sha256': self.vector_sha,
                   'test_behaviors_file': str(self.behaviors),
                   'test_behaviors_sha256': self.behaviors_sha,
                   'user_encoder_state_dict': {
                       k: v.detach().cpu().clone() for k, v in user_encoder.state_dict().items()}}
        with open(path, 'xb') as stream:
            torch.save(payload, stream)
