# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260705-021225-baseline`
- Epochs summarized: 20
- LLM selection ratio: `0.44114285714285717` -> `0.5034285714285714`
- LLM-target minus Human-target MAP5: `-5.687619047619272` -> `-1.3714285714285666`
- Clean table: `results\source_bias_runs\20260705-021225-baseline\analysis\source_bias_summary.csv`
- Dashboard chart: `results\source_bias_runs\20260705-021225-baseline\analysis\source_bias_dashboard.svg`
- MAP5 chart: `results\source_bias_runs\20260705-021225-baseline\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260705-021225-baseline\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
