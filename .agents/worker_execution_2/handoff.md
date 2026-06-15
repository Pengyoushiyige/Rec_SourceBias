# Handoff Report - worker_execution_2

## 1. Observation
- **Predecessor's Fixes**: `src/parameters.py` was already modified to define `--human_news_file` instead of `--HGC_file` (with fallback logic), change `--gpu` default to `'0'`, and resolve relative dataset paths to full directories using `--dataset`.
- **System Resource Limitations**:
  - Running commands with `--gpu 0` failed with pagefile allocation errors:
    ```
    OSError: [WinError 1455] 页面文件太小，无法完成操作 Error loading "E:\Lab\Rec_SourceBias\.venv\lib\site-packages\torch\lib\cudnn_cnn_train64_8.dll" or one of its dependencies.
    ```
    This indicates that CUDA initialization exceeds Windows system commit memory limits.
  - Setting `--gpu -1` forces PyTorch to run on CPU, successfully bypassing CUDA library initialization and resolving the pagefile memory allocation crash.
- **Evaluation Memory Leak**:
  - Run without memory optimizations crashed during testing at batch step `126/160` with exit code `1` due to virtual memory exhaustion.
  - The evaluation routines in `src/test.py` and `src/train_loop.py` did not disable gradient storage during forward passes, causing the BERT computation graph to grow linearly in size.
- **Preprocessing Run**:
  - Command:
    `uv run python src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`
  - Result: Completed successfully in under 2 seconds. The output file `dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv` was created.
- **Training Run**:
  - Command:
    `uv run python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu -1`
  - Result: Completed successfully. Created model checkpoint at `save/sample_model/epoch-1.pt`.
- **Testing Run**:
  - Command:
    `uv run python src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu -1`
  - Result: Completed successfully and printed evaluation metrics for ratios 0.0 to 1.0.
- **Feedback Loop Simulation Run**:
  - Standard Command:
    `uv run python src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1`
  - Debiased Command:
    `uv run python src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1`
  - Result: Both simulations completed successfully without errors.

## 2. Logic Chain
- **GPU Default and Path Handling**:
  - **Observation**: `--HGC_file` was changed to `--human_news_file` in `src/parameters.py`. The path resolution code resolves parameters like `--behaviors_file` to `dataset/Amazon_Beauty_sample/train/behaviors.tsv` and `--human_news_file` to `dataset/Amazon_Beauty_sample/text/...`.
  - **Inference**: The parameter naming bug and dataset path resolution are resolved and verified to work correctly.
- **Bypassing CUDA Windows Crash**:
  - **Observation**: Launching CUDA on Windows caused `OSError: [WinError 1455]`. Setting `--gpu -1` forces CPU usage.
  - **Inference**: Running on CPU successfully avoids loading CUDA cudnn libraries, preventing commit limit errors and allowing python execution to proceed.
- **Fixing Memory Crash during Testing**:
  - **Observation**: Running news vector calculations on BERT without gradient tracking caused a memory leak and crashed testing at step `126/160`.
  - **Inference**: Wrapping evaluation methods (`evaluates`, `get_news2vector`, `get_user2vector`) with `@torch.no_grad()` prevents PyTorch from storing activation graphs, resolving the memory crash completely and allowing evaluation to complete.
- **argparse Debiasing Mode Choices**:
  - **Observation**: The debiased simulation script uses `--debias_type emb_entropy`, but `"emb_entropy"` was not in choices.
  - **Inference**: Adding `"emb_entropy"` to the choices list in `src/parameters.py` resolves choice validation.

## 3. Caveats
- CPU-based training/testing is slower than GPU-based training, but it is extremely reliable on systems with restricted virtual memory pagefiles. On the tiny sample dataset `Amazon_Beauty_sample`, CPU training takes less than 1 minute.
- We assumed that `emb_entropy` was a valid debias type intended for the `--debias_type` argument. This is supported by `emb_entropy` being the default value in `src/parameters.py`.

## 4. Conclusion
- All parameter bugs are successfully resolved.
- Preprocessing, training, evaluation, and feedback loop simulations (standard and debiased) are fully verified and run without errors.
- Memory leaks in evaluation have been fixed by decorating the vector calculations with `@torch.no_grad()`.

## 5. Verification Method
Verify by executing the following commands in the virtual environment:
1. **Preprocessing Verification**:
   `uv run python src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`
2. **Training Verification**:
   `uv run python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu -1`
3. **Testing Verification**:
   `uv run python src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu -1`
4. **Standard Feedback Loop**:
   `uv run python src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1`
5. **Debiased Feedback Loop**:
   `uv run python src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1`
