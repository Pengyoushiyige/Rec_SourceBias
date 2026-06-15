## 2026-06-15T14:51:59Z

Your identity is `worker_documentation_1` and your working directory is `e:\Lab\Rec_SourceBias\.agents\worker_documentation_1`.
Your mission is to write a comprehensive, high-quality documentation file `experiment_guide.md` in the project root directory `e:\Lab\Rec_SourceBias\`.

The guide must be written in Markdown format and cover:
1. **Introduction & Project Overview**: High-level summary of the Rec_SourceBias repository.
2. **Project Directory Structure**: A tree view explaining the purpose of each key file and directory (such as `src/`, `dataset/`, `save/`, and the main source files: `run.py`, `data_preprocess.py`, `train_human.py`, `train_loop.py`, `test.py`, `parameters.py`, `Dataset.py`, `utils.py`, `model/`).
3. **Environmental Setup**: Quick reference to Python virtual environment activation and dependency requirements. Mention the UV cache redirection and Windows compatibility.
4. **Sample Dataset**: Explain the purpose of `dataset/Amazon_Beauty_sample` and how it was created for fast E2E verification of the pipeline.
5. **Exact Command-Line Examples**:
   Provide detailed, copy-pasteable commands for all stages of the pipeline on the sample dataset. Highlight that CPU execution uses `--gpu -1` and Windows multiprocessing pickling limits require `--num_workers 0`:
   - Data preprocessing:
     `python src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`
   - Model training (HGC only):
     `python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu -1 --num_workers 0`
   - Model testing/evaluation:
     `python src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu -1 --num_workers 0`
   - Standard feedback loop simulation:
     `python src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1 --num_workers 0`
   - Debiased feedback loop simulation:
     `python src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1 --num_workers 0`
6. **Key Parameters Guide**:
   Explain key parameters and their roles:
   - `--mode`: Options `['train', 'test', 'loop']`
   - `--dataset`: Folder name under `dataset/`
   - `--model_type`: Models like `BERT4Rec`, `SASRec`, `GRU4Rec`, `LRURec`
   - `--epochs`: Epochs per training run
   - `--loop_epochs`: Epochs in the loop mode simulation
   - `--debias`: Flag to enable debiased loop simulation
   - `--debias_type`: Alignment loss/debias metrics (`emb_entropy`, `dis_user`, `dis_news`, `dis_double`)
   - `--gpu`: GPU device ID (e.g. `'0'`), or `'-1'` for CPU execution
   - `--num_workers`: PyTorch DataLoader worker count (0 for Windows/CPU fallback)
7. **Troubleshooting**:
   Mention issues resolved (like device fallback from `.cuda()` to device-agnostic, pickling errors on Windows, zero-division error in evaluation metrics, and matching human/LLM test evaluation).

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Please write the file `e:\Lab\Rec_SourceBias\experiment_guide.md` directly. Update your `progress.md` in your directory, and write `handoff.md` in your directory once complete.
