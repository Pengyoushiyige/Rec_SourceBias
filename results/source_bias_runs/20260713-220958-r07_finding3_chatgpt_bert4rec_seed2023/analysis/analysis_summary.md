# Rec_SourceBias Analysis Summary

- Run directory: `results\source_bias_runs\20260713-220958-r07_finding3_chatgpt_bert4rec_seed2023`
- Run id: `20260713-220958-r07_finding3_chatgpt_bert4rec_seed2023`
- Return code: `0`
- Metric points: `20`
- Train-loss rows: `100`

## Metric Semantics

- Paper Relative Delta uses `Human Target` / `LLM Target` only.
- Formula: `(Human Target - LLM Target) / mean(Human Target, LLM Target) * 100`.
- Negative values mean `LLM Target` is higher.
- Direct `Human` / `LLM` deltas are diagnostic only.

## Figures

- `figures/figure-01-paper-style-relative-delta.svg`
- `figures/figure-03-ratios.svg`
