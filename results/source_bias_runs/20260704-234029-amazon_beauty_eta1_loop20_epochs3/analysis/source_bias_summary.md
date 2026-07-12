# Source Bias Run Summary

- Run directory: `results\source_bias_runs\20260704-234029-amazon_beauty_eta1_loop20_epochs3`
- Epochs summarized: 20
- LLM selection ratio: `0.524` -> `0.528`
- LLM-target minus Human-target MAP5: `2.900000000000105` -> `1.2533333333332877`
- Clean table: `results\source_bias_runs\20260704-234029-amazon_beauty_eta1_loop20_epochs3\analysis\source_bias_summary.csv`
- Dashboard chart: `results\source_bias_runs\20260704-234029-amazon_beauty_eta1_loop20_epochs3\analysis\source_bias_dashboard.svg`
- MAP5 chart: `results\source_bias_runs\20260704-234029-amazon_beauty_eta1_loop20_epochs3\analysis\map5_by_epoch.svg`
- LLM ratio chart: `results\source_bias_runs\20260704-234029-amazon_beauty_eta1_loop20_epochs3\analysis\llm_ratio_by_epoch.svg`

The source-bias signal is visible when the LLM selection ratio rises across loop epochs and when the LLM-target metrics diverge from the human-target metrics.
