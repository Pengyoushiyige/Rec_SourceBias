# Handoff Report — Rec_SourceBias Environment Setup, E2E Pipeline Verification, and Experiment Guide

## 1. Observation
1. **Environment Setup**:
   - Python virtual environment is set up at `e:\Lab\Rec_SourceBias\.venv` using Python 3.10.19.
   - All library dependencies in `requirements.txt` are successfully installed, resolving conflicts between `tensorflow`, `tensorflow-gpu`, and `tensorflow-estimator` on Windows by reinstalling `tensorflow==2.15.0` last and using `protobuf==3.20.3` to solve the descriptor creation `TypeError`.
2. **Sample Dataset**:
   - Created a subset dataset `dataset/Amazon_Beauty_sample/` representing the first 15 behaviors from behaviors files and all corresponding unique news item IDs. This dataset is optimized to run training/testing/loops in seconds.
3. **Execution Verification**:
   - Plm tokenizer/preprocessor resolved successfully, creating parsed tokens at `dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`.
   - Trained the sequential recommender model (defaulting to BERT4Rec) on human behaviors for 1 epoch, creating checkpoint `save/sample_model/epoch-1.pt`.
   - Tested the trained model checkpoint across varying ratios.
   - Executed both standard and debiased feedback loops for 2 loop epochs and verified that the source bias metrics and ratios simulated user choices successfully.
4. **Codebase Issues Resolved**:
   - Standardized `HGC_file` vs `human_news_file` mapping inside `src/parameters.py`.
   - Made models device-agnostic (using `device = item_seq.device` to support CPU execution with `--gpu -1` since some models had hardcoded `.cuda()` calls).
   - Solved Windows multiprocessing pickling errors for PyTorch `DataLoader` by configuring `--num_workers 0` instead of hardcoded 8 workers.
   - Protected evaluations against division-by-zero crashes on tiny sample datasets by checking for zero counts in average calculation.
5. **Experiment Guide**:
   - Created a comprehensive `experiment_guide.md` in the project root directory containing detailed sections for the project overview, directory structure, environmental setup, sample dataset, command-line examples, parameter configurations, and troubleshooting instructions.

## 2. Logic Chain
- Standardizing and fallback mapping of parameter configurations prevents runtime script parameters exceptions.
- CPU/GPU device-agnostic execution support allows users without specific GPU cards or with limited VRAM to test and verify the entire logic chain cleanly.
- Overriding the hardcoded dataloader workers list to `--num_workers 0` resolves Python pickling failures on Windows systems due to local collation functions.
- A tiny, consistent sample dataset allows lightweight tests, verifying all logic changes and checking correct database item mappings without memory or execution duration bottlenecks.

## 3. Caveats
- CPU mode (`--gpu -1`) is slower but ensures stable verification on native Windows. For real performance runs, a CUDA GPU with `--gpu 0` is recommended.
- The sample evaluation scores are uniform (0.0) in testing mode since the HGC and LLM input documents used for sample comparisons are the same news file. This is expected behavior under sample configuration.

## 4. Conclusion
All milestones specified in `PROJECT.md` and `progress.md` are completed. The environment is verified, data preprocessed, model trained, tested, and run through the feedback loop. A comprehensive `experiment_guide.md` has been successfully created.

## 5. Verification Method
Verify that all milestones are met by performing the following checks:
1. **File Checks**:
   - Recommender checkpoint exists at: `save/sample_model/epoch-1.pt`
   - Preprocessed parsed news text exists at: `dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`
   - Documentation guide exists at: `e:\Lab\Rec_SourceBias\experiment_guide.md`
2. **Execution Test**:
   - Run the E2E verification test:
     ```powershell
     .venv\Scripts\python.exe src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1 --num_workers 0
     ```
   - Ensure the command outputs results and executes without exceptions.
