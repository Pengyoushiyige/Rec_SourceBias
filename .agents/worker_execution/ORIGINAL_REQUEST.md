## 2026-06-15T13:41:38Z
You are a Worker subagent. Your identity is `worker_execution` and your working directory is `e:\Lab\Rec_SourceBias\.agents\worker_execution`.

Your tasks are:
1. Generate a sample dataset by running:
   `python .agents/teamwork_preview_explorer_exploration/create_sample_dataset.py --n 15`
   Verify that the folder `dataset/Amazon_Beauty_sample` and its subfolders/files are created.

2. Resolve a parameter naming bug in the codebase:
   - Notice that `src/parameters.py` defines `--HGC_file` (line 49) but the rest of the codebase (`src/train_human.py`, `src/train_loop.py`, `src/test.py`) accesses `args.human_news_file`.
   - Edit `src/parameters.py` to define `--human_news_file` instead of `--HGC_file`, or map `args.human_news_file = args.HGC_file`.
   - Also notice that the default `--gpu` is `'3'` which might cause errors on systems with fewer GPUs. Change the default gpu to `'0'` (or check if `torch.cuda.is_available()` and make sure it uses an available GPU).

3. Verify data preprocessing on the sample dataset:
   - Run the preprocessing script:
     `python src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv`
   - Verify that the target file `news_parsed_bert-base-uncased_test.tsv` is successfully created.

4. Verify training (`--mode train`):
   - Run the training script on the sample dataset using the virtual environment:
     `python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu 0`
   - Verify that the model checkpoint (e.g. `save/sample_model/epoch-1.pt`) is successfully saved. Note: create the checkpoint folder first if it does not exist or make sure it gets created.

5. Verify testing (`--mode test`):
   - Run the evaluation script using the saved checkpoint:
     `python src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu 0`
   - Verify it finishes successfully and prints metrics.

6. Verify feedback loop simulation (`--mode loop`):
   - Run standard feedback loop training:
     `python src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu 0`
   - Run debiased feedback loop training:
     `python src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu 0`
   - Verify that both run without errors.

7. Update your progress log in `e:\Lab\Rec_SourceBias\.agents\worker_execution\progress.md` after each step.
8. Document all commands, modifications, and execution outputs in a detailed handoff report `e:\Lab\Rec_SourceBias\.agents\worker_execution\handoff.md`.

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.
