# Handoff Report — Exploration of Rec_SourceBias and Amazon Beauty Dataset

## 1. Observation

### Codebase Structure and File Roles
- **`src/data_preprocess.py`**: Reads raw news TSV files and tokenizes their abstracts using `transformers.AutoTokenizer`.
- **`src/run.py`**: The main entry point. Sets random seeds, determines the mode (`train`, `test`, `loop`), and calls the corresponding trainer or tester class.
- **`src/train_human.py`**: Trains the recommendation model using human-generated content (HGC).
- **`src/train_loop.py`**: Simulates the feedback loop by alternating training on two halves of behaviors, generating recommendation vectors, and updating/debiasing the user profiles over multiple loop epochs.
- **`src/test.py`**: Tests the trained model checkpoint across varying ratios of LLM-generated vs human-generated content.
- **`src/utils.py`**: Utility functions for logging, checkpoints, loss computation, model factory lookup (`BERT4Rec`, `SASRec`, `GRU4Rec`, `LRURec`), and evaluation metrics.
- **`src/parameters.py`**: Defines all command-line arguments and configuration parameters.
- **`src/Dataset.py`**: Custom PyTorch Datasets.
  - `BaseDataset` (lines 38-164): Loads behavior records and dynamically blends human/LLM-generated news candidates.
  - `NewsDataset` (lines 165-203): Loads parsed news files and parses the token arrays.
  - `UserDataset` (lines 204-254): Compiles user history sequences for profile encoding.
  - `BehaviorsDataset` (lines 255-315): Loads behavior evaluation logs.

### Dataset Files (under `dataset/Amazon_Beauty/`)
1. **`train/behaviors.tsv`**, **`train/behaviors1.tsv`**, **`train/behaviors2.tsv`**, **`dev/behaviors.tsv`**:
   - Tab-separated values file without header.
   - Format: `[impression_id]\t[user_id]\t[timestamp]\t[click_history]\t[impressions]`
   - Example line (`train/behaviors.tsv` line 1):
     `326	A13CZ95VU0QF3U	1399248000	B000V58ANS B001782OO4 B000RQ1DAI B000CECSAO B0019TYZ66	B00E56I6M4-1 B00147FGJ8-0 B001KFDHZE-0 B002C0338W-0 B0013ZCPIG-0`
     - `click_history`: Space-separated item IDs of items clicked in the past.
     - `impressions`: Space-separated item candidate IDs with `-label` suffix where `-1` represents a click/positive interaction, and `-0` represents non-clicked/negative.
2. **`text/news.tsv`**, **`text/news_llama.tsv`**, **`text/news_gemini-pro.tsv`**, etc.:
   - Raw news files, tab-separated values without header.
   - Format: `[id]\t\t\t\t[abstract]` (Column 0: item ID, Columns 1-3: empty strings, Column 4: abstract text description).
3. **`text/news_parsed_bert-base-uncased.tsv`**:
   - Output of `data_preprocess.py`. Tab-separated with header:
     `id\tabstract_input_ids\tabstract_token_type_ids\tabstract_attention_mask`
   - Example line:
     `7806397051	[101, 2019, 4866, ...]	[0, 0, 0, ...]	[1, 1, 1, ...]`

### Identified Code Issues / Dependencies
- **Hardcoded `.cuda()` Calls**:
  - `src/model/BERT4Rec.py` (lines 67, 74): `.cuda()` is hardcoded.
  - `src/model/SASRec.py` (lines 63, 70): `.cuda()` is hardcoded.
  - `src/model/click_model/ncm.py` (lines 57, 63): `.cuda()` is hardcoded.
  - *Implication*: Running the code on a CPU-only environment will result in a runtime error unless a CUDA device is present or these lines are refactored to `.to(device)`.

---

## 2. Logic Chain

1. **Missing ID Lookup Risk**:
   - In `Dataset.py`'s `NewsDataset`, the parsed news file is read and loaded into a dictionary (`self.news2dict`).
   - During training, the user history items and candidates are looked up: `self.human_text2vector[item_id]` or `self.llm_text2vector[item_id]`.
   - If any item ID present in `behaviors.tsv` (in click history or impressions list) is not found in the news files, a `KeyError` will be thrown.
2. **Consistency Constraint**:
   - To make a valid sample dataset, we must sample a small subset of rows from behaviors files, extract the union of all item IDs appearing in their click history and impression candidate lists, and keep only those item IDs in the raw and parsed news files.
3. **Execution Speed**:
   - The full dataset contains ~12,000 news articles and thousands of behaviors.
   - Restricting behaviors to 15 records limits the unique item IDs to ~100. This dramatically speeds up vocabulary loading, tokenization, model user encoding, and evaluation loops to run in <2 seconds.

---

## 3. Caveats

- **CUDA Dependency**: As highlighted above, SASRec and BERT4Rec models have hardcoded `.cuda()` calls. Thus, the verification of the sample dataset still requires a CUDA-enabled GPU device, or the implementer must replace `.cuda()` calls with `.to(device)`.
- **Pre-Parsed Files Alignment**: The pre-parsed news files in the sample (like `news_parsed_bert-base-uncased.tsv`) are pre-tokenized using `bert-base-uncased`. If a different model is used (e.g. `roberta-base`), `data_preprocess.py` must be executed to regenerate tokenizations for the sample.

---

## 4. Conclusion

We have analyzed the codebase structure and dataset formats. We have created a standalone Python script `.agents/teamwork_preview_explorer_exploration/create_sample_dataset.py` that automates the generation of a consistent, lightweight sample dataset under `dataset/Amazon_Beauty_sample/`. 

### Summary of Parameters in `parameters.py`
| Parameter Name | Type / Default | Role / Description |
|---|---|---|
| `--mode` | `str` / `test` | Runtime mode: `train` (HGC only), `test` (mixed HGC/LLM evaluation), `loop` (feedback loop). |
| `--dataset` | `str` / `Amazon_Beauty` | Dataset folder name. |
| `--model_type` | `str` / `BERT4Rec` | Recommendation model type (`BERT4Rec`, `SASRec`, `GRU4Rec`, `LRURec`). |
| `--batch_size` | `int` / `256` | Minibatch size for training and testing. |
| `--shuffle_buffer_size` | `int` / `10000` | Buffer size for dataset shuffling. |
| `--num_workers` | `int` / `4` | Number of parallel data loader worker processes. |
| `--log_steps` | `int` / `100` | Logging frequency (number of steps). |
| `--epochs` | `int` / `3` | Number of training epochs per training iteration. |
| `--lr` | `float` / `1e-3` | Learning rate. |
| `--weight_decay` | `float` / `1e-5` | L2 weight decay regularization coefficient. |
| `--news_attributes` | `list` / `['abstract']` | News features to use as input. |
| `--num_words_abstract` | `int` / `100` | Maximum token limit for abstract text. |
| `--user_log_length` | `int` / `10` | Maximum history sequence length for a user. |
| `--word_embedding_dim` | `int` / `768` | Hidden size of the text representation embeddings. |
| `--user_log_mask` | `bool` / `True` | Whether to mask padded history entries. |
| `--drop_rate` | `float` / `0.2` | Dropout probability. |
| `--load_ckpt_name` | `str` / `epoch-3.pt` | Model checkpoint file name to load. |
| `--gpu` | `str` / `3` | GPU device ID (passed to `CUDA_VISIBLE_DEVICES`). |
| `--plm_model` | `str` / `bert-base-uncased` | HuggingFace pretrained model for tokenization/embeddings. |
| `--llm_model` | `str` / `llama` | Name of the LLM generator. |
| `--negative_sampling_ratio`| `int` / `1` | Ratio of negative samples to positive samples in training. |
| `--behaviors_file` | `str` / `behaviors.tsv` | Behaviors filename for training. |
| `--behaviors_file1` | `str` / `behaviors1.tsv` | First half behaviors file for loop training. |
| `--behaviors_file2` | `str` / `behaviors2.tsv` | Second half behaviors file for loop training. |
| `--test_behaviors_file` | `str` / `behaviors.tsv` | Behaviors filename for testing. |
| `--llm_news_file` | `str` / `news_llama_parsed_bert-base-uncased.tsv` | Parsed LLM-generated news articles file path. |
| `--llm_rewirte_news_file` | `str` / `news_llama_rewrite_bert-base-uncased.tsv` | Parsed rewritten news file path (for debiasing). |
| `--debias` | `bool` / `False` | Enable source debiasing in feedback loop training. |
| `--debias_type` | `str` / `emb_entropy` | Type of debiasing objective to apply. |
| `--disturb_ratio` | `float` / `0.5` | Ratio for history representation perturbation. |
| `--eta` | `float` / `-1` | Position bias parameter for click simulation. |
| `--loop_epochs` | `int` / `10` | Number of feedback loop epochs. |
| `--user_loss` | `float` / `0.01` | Weight for user profile alignment loss. |
| `--news_loss` | `float` / `1` | Weight for news description alignment loss. |
| `--ckpt_dir` | `str` / `""` | Directory to save checkpoints. |

---

## 5. Verification Method

To verify the correct creation and run validity of the sample dataset:

1. **Generate the Sample Dataset**:
   Run the generated Python script to create the sample dataset:
   ```powershell
   python .agents/teamwork_preview_explorer_exploration/create_sample_dataset.py --n 15
   ```
   *Expected Output*: Creation of the `dataset/Amazon_Beauty_sample/` directory structure containing `train/`, `dev/`, and `text/` files, each trimmed to only include the first 15 behaviors and their referenced item IDs.

2. **Verify Training Runs successfully**:
   Execute the training command using the sample dataset (using a CPU or GPU machine with low batch size/epochs):
   ```powershell
   python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4
   ```
   *Pass Condition*: The code loads datasets successfully and completes training epoch 1 without throwing a `KeyError`.
