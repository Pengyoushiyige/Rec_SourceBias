# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260705-015434-amazon_beauty_standard_distinct_gpu0_loop10_paper`
- Epochs summarized: 10
- LLM selection ratio: `0.44114285714285717` -> `0.5308571428571428`
- LLM-target minus Human-target MAP5: `-5.687619047619272` -> `-0.9742857142857027`
- Clean table: `results\source_bias_runs\20260705-015434-amazon_beauty_standard_distinct_gpu0_loop10_paper\analysis\source_bias_summary.csv`
- Dashboard chart: `results\source_bias_runs\20260705-015434-amazon_beauty_standard_distinct_gpu0_loop10_paper\analysis\source_bias_dashboard.svg`
- MAP5 chart: `results\source_bias_runs\20260705-015434-amazon_beauty_standard_distinct_gpu0_loop10_paper\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260705-015434-amazon_beauty_standard_distinct_gpu0_loop10_paper\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
