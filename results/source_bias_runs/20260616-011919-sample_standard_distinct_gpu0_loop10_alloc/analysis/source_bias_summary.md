# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260616-011919-sample_standard_distinct_gpu0_loop10_alloc`
- Epochs summarized: 10
- LLM selection ratio: `0.6` -> `0.6`
- Clean table: `results\source_bias_runs\20260616-011919-sample_standard_distinct_gpu0_loop10_alloc\analysis\source_bias_summary.csv`
- MAP5 chart: `results\source_bias_runs\20260616-011919-sample_standard_distinct_gpu0_loop10_alloc\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260616-011919-sample_standard_distinct_gpu0_loop10_alloc\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
