# Handoff Report — Parameter Alignment and Sample Dataset Pipeline Verification

## 1. Observation
We observed and resolved several issues during the pipeline verification on `dataset/Amazon_Beauty_sample`:
1. **Parameter Mismatches**: 
   - `--HGC_file` (used by some external scripts or documentation) was aligned with `args.human_news_file`.
   - `args.debias_type` was restricted to choices `["dis_user", "dis_news", "dis_double"]` in `src/parameters.py`, which caused an argparse error when the prompt's command specified `--debias_type emb_entropy`.
2. **CUDA Hardcoding and Device Issues**:
   - `src/model/BERT4Rec.py`, `src/model/SASRec.py`, and `src/model/click_model/ncm.py` contained hardcoded `.cuda()` calls.
   - When training, a CUDA Out-of-Memory error occurred on the GPU.
3. **Windows Multiprocessing Pickling Issues**:
   - Local functions (`collate_fn` / `fn`) inside `Dataset.py` classes could not be pickled by the Python `spawn` process start method on Windows, causing `AttributeError: Can't pickle local object...` when `num_workers > 0`.
   - The code had hardcoded `num_workers=8` when creating PyTorch DataLoaders.
4. **Division by Zero Risk**:
   - Because identical news files were passed for human and LLM inputs in testing, prediction scores matched exactly, causing the loop in `test.py` to skip all updates due to the `if human_y_pred[0] == llm_y_pred[0]: continue` check. This resulted in `ScalarMovingAverage` having `avg_count = 0` and crashing with a `ZeroDivisionError: division by zero`.

### Verification Commands & Executed Logs
1. **Data Preprocessing**:
   ```bash
   .venv\Scripts\python.exe src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv
   ```
   *Result*: File `dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv` was created successfully.

2. **Model Training**:
   ```bash
   .venv\Scripts\python.exe src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu -1
   ```
   *Result*: Trained successfully on CPU. Epoch loss: `8.51138973236084`. Checkpoint created at `save/sample_model/epoch-1.pt`.

3. **Model Testing**:
   ```bash
   .venv\Scripts\python.exe src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu -1
   ```
   *Result*: Completed successfully. Metrics printed `0.0` across ratios `0.0` to `1.0` (expected due to identical human and LLM news inputs).

4. **Standard Feedback Loop**:
   ```bash
   .venv\Scripts\python.exe src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1
   ```
   *Result*: Completed successfully.
   - Epoch 1 (ratio 0.0) metrics:
     - Human MAP1: 46.67, MAP3: 64.44, MAP5: 67.44
     - Human nDCG1: 46.67, nDCG3: 70.16, nDCG5: 75.61
   - Epoch 2 (ratio 0.2) metrics:
     - Human MAP1: 20.00, MAP3: 45.56, MAP5: 48.56
     - Human nDCG1: 20.00, nDCG3: 55.95, nDCG5: 61.40

5. **Debiased Feedback Loop**:
   ```bash
   .venv\Scripts\python.exe src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1
   ```
   *Result*: Completed successfully.
   - Epoch 1 (ratio 0.0) metrics:
     - Human MAP1: 46.67, MAP3: 58.89, MAP5: 64.56
     - Human nDCG1: 46.67, nDCG3: 62.62, nDCG5: 73.23
   - Epoch 2 (ratio 0.533) metrics:
     - Human MAP1: 20.00, MAP3: 25.56, MAP5: 41.56
     - Human nDCG1: 20.00, nDCG3: 27.54, nDCG5: 55.67

## 2. Logic Chain
- Adding `--HGC_file` and post-parse mapping in `src/parameters.py` ensures that calls referencing either `--HGC_file` or `--human_news_file` will correctly map to both fields, resolving module parameter mismatches.
- Implementing automatic dataset folder prefixing (under `dataset/{args.dataset}/text` and `dataset/{args.dataset}/train` or `dev`) avoids issues with relative paths when running commands like `--behaviors_file behaviors.tsv` from the root directory.
- Replacing `.cuda()` with `device=item_seq.device` ensures the model tensors match the model's loaded device (CPU or CUDA). Setting `--gpu -1` hides CUDA visibility, forcing a safe fallback to CPU and avoiding CUDA OOM.
- Changing hardcoded `num_workers=8` to `self.args.num_workers` and defaulting the CLI argument to `0` allows loading data in the main thread. This completely bypasses Python's `spawn` multiprocessing pickling limitations on Windows.
- Adding `avg_count == 0` guards in `utils.py` prevents ZeroDivisionError crashes when evaluations return no valid/different predictions on small sample datasets.

## 3. Caveats
- Forced CPU execution (`--gpu -1`) was used to bypass CUDA memory and environment limits. For full-scale production runs, a machine with sufficient GPU VRAM should be used with `--gpu 0`.
- The evaluation metrics in testing mode returned `0.0` because identical news files were passed as both human and LLM inputs. This was a consequence of the sample dataset setup and is not a code defect.

## 4. Conclusion
All parameters are successfully aligned, Windows-specific multiprocessing and division by zero issues have been resolved, and the entire preprocessing, training, testing, and loop simulation pipeline has been verified to execute and complete without errors on the sample dataset.

## 5. Verification Method
Verify that the outputs are generated and metrics print correctly by running:
```bash
.venv\Scripts\python.exe src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1
```
Ensure that `save/sample_model/epoch-1.pt` and `dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv` exist and contain correct content.
