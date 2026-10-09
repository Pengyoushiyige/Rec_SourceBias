"""Conditional positive-click policies; simulator parameters do not learn."""
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence
import numpy as np


@dataclass(frozen=True)
class ClickConfig:
    click_model: str = 'rank'
    eta: float = -1
    sacm_alpha_mode: str = 'user_beta'
    sacm_alpha_value: float = 0
    sacm_alpha_a: float = 2
    sacm_alpha_b: float = 5
    sacm_cue_mode: str = 'beta'
    sacm_hgc_cue_a: float = 2
    sacm_hgc_cue_b: float = 5
    sacm_aigc_cue_a: float = 5
    sacm_aigc_cue_b: float = 2

    @classmethod
    def from_args(cls, args: Any) -> 'ClickConfig':
        return cls(**{name: getattr(args, name, field.default)
                      for name, field in cls.__dataclass_fields__.items()})

    def __post_init__(self) -> None:
        if self.click_model not in ('rank', 'pbm_probability', 'sacm_static'):
            raise ValueError('Unknown click_model')
        if not math.isfinite(self.eta) or (self.click_model != 'rank' and self.eta < 0):
            raise ValueError('Probability click models require finite eta >= 0 (use --eta 1)')
        if self.sacm_alpha_mode not in ('user_beta', 'global'):
            raise ValueError('Unknown sacm_alpha_mode')
        if self.sacm_cue_mode not in ('beta', 'disabled', 'binary'):
            raise ValueError('Unknown sacm_cue_mode')
        if not math.isfinite(self.sacm_alpha_value) or not 0 <= self.sacm_alpha_value <= 1:
            raise ValueError('sacm_alpha_value must be in [0, 1]')
        for name in ('sacm_alpha_a', 'sacm_alpha_b', 'sacm_hgc_cue_a',
                     'sacm_hgc_cue_b', 'sacm_aigc_cue_a', 'sacm_aigc_cue_b'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(name + ' must be finite and positive')


class ClickPolicy:
    def __init__(self, config: ClickConfig, seed: int) -> None:
        self.config = config
        self.seed = int(seed)
        self.alphas = {}
        self.cues = {}

    def _rng(self, namespace: str, key: Any) -> np.random.Generator:
        payload = json.dumps([self.seed, namespace, key], ensure_ascii=False,
                             separators=(',', ':'), default=str).encode('utf-8')
        return np.random.default_rng(int.from_bytes(hashlib.sha256(payload).digest()[:16], 'big'))

    def alpha(self, user_id: Any) -> float:
        key = str(user_id)
        if key not in self.alphas:
            c = self.config
            self.alphas[key] = (c.sacm_alpha_value if c.sacm_alpha_mode == 'global'
                                else float(self._rng('alpha', key).beta(c.sacm_alpha_a, c.sacm_alpha_b)))
        return self.alphas[key]

    def cue(self, candidate_id: str) -> float:
        key = str(candidate_id)
        source = self.source(key)
        if key not in self.cues:
            c = self.config
            if c.sacm_cue_mode == 'disabled':
                value = 0.0
            elif c.sacm_cue_mode == 'binary':
                value = float(source == 'llm')
            else:
                a, b = ((c.sacm_hgc_cue_a, c.sacm_hgc_cue_b) if source == 'human'
                        else (c.sacm_aigc_cue_a, c.sacm_aigc_cue_b))
                value = float(self._rng('cue', key).beta(a, b))
            self.cues[key] = value
        return self.cues[key]

    @staticmethod
    def source(candidate_id: str) -> str:
        source = str(candidate_id).rsplit('-', 1)[-1]
        if source not in ('human', 'llm'):
            raise ValueError('Candidate ID must end in -human or -llm: ' + str(candidate_id))
        return source

    def select(self, scores: Sequence[float], labels: Sequence[int],
               candidate_ids: Sequence[str], user_id: Any,
               event_key: Any) -> tuple[int, np.ndarray, Mapping[str, Any]]:
        try:
            return self._select(scores, labels, candidate_ids, user_id, event_key)
        except (ValueError, TypeError, IndexError) as error:
            raise ValueError(f'Click impression {event_key!r}: {error}') from error

    def _select(self, scores: Sequence[float], labels: Sequence[int],
                candidate_ids: Sequence[str], user_id: Any,
                event_key: Any) -> tuple[int, np.ndarray, Mapping[str, Any]]:
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels)
        ids = [str(i) for i in candidate_ids]
        if scores.ndim != 1 or labels.shape != scores.shape or len(ids) != len(scores):
            raise ValueError('Candidate arrays must be aligned one-dimensional arrays')
        if not np.all(np.isfinite(scores)) or not np.all(np.isin(labels, [0, 1])):
            raise ValueError('Scores must be finite and labels binary')
        sources = [self.source(i) for i in ids]
        if not np.any(labels == 1):
            raise ValueError('No positive candidate')
        c = self.config
        order = (np.argsort(scores)[::-1] if c.click_model == 'rank'
                 else np.lexsort((np.asarray(ids), -scores)))
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        alpha = self.alpha(user_id) if c.click_model == 'sacm_static' else 0.0
        cues = np.asarray([self.cue(i) for i in ids]) if c.click_model == 'sacm_static' else np.zeros(len(ids))
        positives = order[labels[order] == 1]
        probabilities = np.zeros(len(scores))
        if c.click_model == 'rank' and c.eta == -1:
            selected = int(positives[0])
            probabilities[selected] = 1.0
        elif c.click_model == 'rank':
            weights = np.asarray([pow(1 / int(ranks[i]), c.eta) for i in positives])
            p = weights / weights.sum()
            selected = int(positives[np.random.choice(range(len(positives)), p=p)])
            probabilities[positives] = p
        else:
            weights = (labels == 1) * np.power(ranks.astype(float), -c.eta)
            if c.click_model == 'sacm_static':
                weights = weights * (1 - alpha * cues)
            total = weights[order].sum()
            if not np.isfinite(total) or total <= 0:
                raise ValueError('Total click weight is zero or non-finite')
            probabilities = weights / total
            # Canonical ranking also makes tied candidates independent of source ordering.
            draw = self._rng('click', event_key).random()
            supported = order[probabilities[order] > 0]
            position = min(int(np.searchsorted(np.cumsum(probabilities[supported]), draw, side='right')), len(supported) - 1)
            selected = int(supported[position])
        diagnostics = dict(candidate_ids=ids, sources=sources, scores=scores.tolist(),
                           labels=labels.tolist(), ranks=ranks.tolist(), cues=cues.tolist(),
                           alpha=float(alpha), probabilities=probabilities.tolist(),
                           selected_index=selected, user_id=str(user_id), click_model=c.click_model)
        return selected, probabilities, diagnostics
