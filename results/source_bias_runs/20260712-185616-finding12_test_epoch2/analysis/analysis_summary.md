# Rec_SourceBias Analysis Summary

- Run directory: `results\source_bias_runs\20260712-185616-finding12_test_epoch2`
- Run id: `20260712-185616-finding12_test_epoch2`
- Return code: `0`
- Metric points: `11`
- Train-loss rows: `0`

## Metric Semantics

- Paper Relative Delta uses `Human Target` / `LLM Target` only.
- Formula: `(Human Target - LLM Target) / mean(Human Target, LLM Target) * 100`.
- Negative values mean `LLM Target` is higher.
- Direct `Human` / `LLM` deltas are diagnostic only.

## Figures

- `figures/figure-01-paper-style-relative-delta.svg`
- `figures/figure-02-target-gap-trajectory.svg`
