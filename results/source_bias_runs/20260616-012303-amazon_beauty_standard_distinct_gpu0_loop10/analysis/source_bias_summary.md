# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260616-012303-amazon_beauty_standard_distinct_gpu0_loop10`
- Epochs summarized: 10
- LLM selection ratio: `0.5468571428571428` -> `0.652`
- LLM-target minus Human-target MAP5: `5.9514285714287` -> `12.68095238095231`
- Clean table: `results\source_bias_runs\20260616-012303-amazon_beauty_standard_distinct_gpu0_loop10\analysis\source_bias_summary.csv`
- Dashboard chart: `results\source_bias_runs\20260616-012303-amazon_beauty_standard_distinct_gpu0_loop10\analysis\source_bias_dashboard.svg`
- MAP5 chart: `results\source_bias_runs\20260616-012303-amazon_beauty_standard_distinct_gpu0_loop10\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260616-012303-amazon_beauty_standard_distinct_gpu0_loop10\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
