# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260705-002830-beauty_reproduce_standard_gpu0_bs64`
- Epochs summarized: 20
- LLM selection ratio: `0.44971428571428573` -> `0.6194285714285714`
- LLM-target minus Human-target MAP5: `-5.497142857142968` -> `8.920952380952297`
- Clean table: `results\source_bias_runs\20260705-002830-beauty_reproduce_standard_gpu0_bs64\analysis\source_bias_summary.csv`
- Dashboard chart: `results\source_bias_runs\20260705-002830-beauty_reproduce_standard_gpu0_bs64\analysis\source_bias_dashboard.svg`
- MAP5 chart: `results\source_bias_runs\20260705-002830-beauty_reproduce_standard_gpu0_bs64\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260705-002830-beauty_reproduce_standard_gpu0_bs64\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
