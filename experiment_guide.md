# Rec_SourceBias: Experiment Guide

This guide provides a comprehensive overview of the **Rec_SourceBias** repository, detailed instructions for setting up the environment, description of the sample dataset, command-line examples for E2E validation, a guide to key parameters, and troubleshooting advice.

---

## 1. Introduction & Project Overview

**Rec_SourceBias** is the official implementation for the paper *Exploring the Escalation of Source Bias in User, Data, and Recommender System Feedback Loop* (SIGIR 2025). 

Modern recommender systems operate in a feedback loop: models are trained on user interaction data, which generates recommendations that in turn influence future user behavior. With the rise of AI-generated content (AIGC), recommender systems are increasingly exposed to a mix of Human-Generated Content (HGC) and LLM-generated/rewritten content. This repository simulates and models how **source bias**—the preference or systemic bias towards a particular source (HGC vs. AIGC)—propagates and escalates over multiple recommendation cycles.

Key goals and components of the pipeline:
- **Sequential Recommendation Modeling**: Employs state-of-the-art sequential recommenders (e.g., BERT4Rec, SASRec, GRU4Rec, LRURec) to capture user click histories.
- **Feedback Loop Simulation**: Runs multi-stage simulations where models generate recommendations, simulated users interact with them based on their historical preferences, and new interaction data is logged to retrain the recommender model.
- **Debiasing & Alignment**: Evaluates debiasing methods (such as entropy regularization and contrastive alignment) to mitigate the escalation and dominance of AIGC source bias.

---

## 2. Project Directory Structure

Below is the directory structure of the repository, explaining the purpose of each key folder and file:

```text
.
├── dataset/                     # Contains behavioral logs and item textual metadata
│   ├── Amazon_Beauty/           # Beauty dataset files
│   ├── Amazon_Health/           # Health dataset files
│   ├── Amazon_Sports/           # Sports dataset files
│   └── Amazon_Beauty_sample/    # Small dataset sample for fast E2E pipeline verification
│       ├── train/               # Sample behavioral logs for training and loop simulation
│       │   ├── behaviors.tsv
│       │   ├── behaviors1.tsv
│       │   └── behaviors2.tsv
│       ├── dev/                 # Sample behavioral logs for validation/dev evaluation
│       │   └── behaviors.tsv
│       └── text/                # Sample item text metadata (HGC, LLM, parsed versions)
│           ├── news.tsv
│           ├── news_Mistral-7B-Instruct-v0.2.tsv
│           ├── news_gemini-pro.tsv
│           ├── news_gpt-3.5-turbo-0613.tsv
│           ├── news_gpt-3.5-turbo-0613-rewrite.tsv
│           ├── news_llama.tsv
│           └── news_parsed_bert-base-uncased.tsv
├── save/                        # Directory where model checkpoints (.pt files) are saved
├── src/                         # Core Python source code directory
│   ├── model/                   # Implementations of sequential recommender models
│   │   ├── BERT4Rec.py          # BERT-based sequential recommendation model
│   │   ├── GRU4Rec.py           # GRU-based sequential recommendation model
│   │   ├── LRURec.py            # Linear Recurrent Unit (LRU) sequential recommendation model
│   │   ├── SASRec.py            # Self-Attention based sequential recommendation model
│   │   └── click_model/         # Simulated user click behavior models (e.g., ncm.py)
│   ├── run.py                   # Unified command-line entry point
│   ├── data_preprocess.py       # Plm-based text parser/tokenizer for items/news
│   ├── train_human.py           # Script to train recommenders solely on human interactions
│   ├── train_loop.py            # Code for multi-stage feedback loop simulation and debiasing
│   ├── test.py                  # Evaluates model performance under varying HGC/AIGC ratios
│   ├── Dataset.py               # Custom PyTorch Dataset and DataLoader code
│   ├── parameters.py            # Command-line argument parser and resolution rules
│   ├── utils.py                 # Utility helper functions (checkpoints, models mapping, logging)
│   └── metrics.py               # Evaluates ranking metrics (nDCG and MAP at k=1, 3, 5)
├── requirements.txt             # List of python package dependencies
├── PROJECT.md                   # Project architecture, contracts, and milestones tracking
└── README.md                    # Original quick start and citation details
```

---

## 3. Environmental Setup

### Virtual Environment Activation
To isolate dependencies, it is recommended to create and activate a Python virtual environment:

- **Windows (PowerShell)**:
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux/macOS**:
  ```bash
  source .venv/bin/activate
  ```

### Installing Dependencies
Install the required packages using `pip` or the faster `uv` package manager:

```bash
# Using standard pip
pip install -r requirements.txt

# Or using uv (faster resolution)
uv pip install -r requirements.txt
```

#### UV Cache Redirection
When using `uv` on Windows or inside environments with space constraints on the primary drive, you can redirect the cache directory using the `UV_CACHE_DIR` environment variable:
```powershell
# PowerShell
$env:UV_CACHE_DIR="e:\.uv_cache"

# Command Prompt
set UV_CACHE_DIR=e:\.uv_cache
```

#### Windows Compatibility Notes
- Ensure your Python version is **3.10.x** (specifically tested on `3.10.13` or similar compatible 3.10 releases).
- GPU execution requires CUDA-compatible PyTorch binaries. If not using a GPU, command arguments must fallback to CPU execution (`--gpu -1`).
- Windows-specific multiprocessing serialization and pickling limitations require setting `--num_workers 0` in PyTorch DataLoader configurations.

---

## 4. Sample Dataset

To avoid processing millions of logs and items during debugging or workflow verification, a tiny E2E test dataset is provided at `dataset/Amazon_Beauty_sample`.

### Purpose
- **Speed**: Allows testing preprocessing, training, testing, and feedback loop code within a minute.
- **Verification**: Verifies model convergence, directory/path resolution rules, and logging structure.
- **Simplicity**: Provides smaller behavior logs (`behaviors1.tsv` and `behaviors2.tsv`) to simulate feedback loop splits without the need for large-scale prep.

---

## 5. Exact Command-Line Examples

Use these copy-pasteable commands to run each stage of the pipeline on the sample dataset. 

*Note: For Windows/CPU compatibility and multiprocessing stability, CPU execution uses `--gpu -1` and PyTorch worker limits require `--num_workers 0`.*

### A. Data Preprocessing
Tokenizes the news/item text abstract using BERT and outputs parsed IDs and masks:
```bash
python src/data_preprocess.py --text_source dataset/Amazon_Beauty_sample/text/news.tsv --text_parsed_target dataset/Amazon_Beauty_sample/text/news_parsed_bert-base-uncased_test.tsv
```

### B. Model Training (HGC Only)
Trains the recommender model solely on human-generated content behavior logs and saves the checkpoint to `save/sample_model/`:
```bash
python src/run.py --mode train --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --ckpt_dir save/sample_model --epochs 1 --batch_size 4 --gpu -1 --num_workers 0
```

### C. Model Testing / Evaluation
Evaluates the trained checkpoint over varying mixtures of human and LLM-generated content:
```bash
python src/run.py --mode test --dataset Amazon_Beauty_sample --behaviors_file behaviors.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --load_ckpt_name save/sample_model/epoch-1.pt --batch_size 4 --gpu -1 --num_workers 0
```

### D. Standard Feedback Loop Simulation
Simulates the multi-epoch recommendation feedback loop, using two behavior splits to model user clicks updates over time:
```bash
python src/run.py --mode loop --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1 --num_workers 0
```

### E. Debiased Feedback Loop Simulation
Simulates the feedback loop with debiased alignment (using `--debias` and `--debias_type emb_entropy` along with LLM-rewritten news text):
```bash
python src/run.py --mode loop --debias --debias_type emb_entropy --dataset Amazon_Beauty_sample --behaviors_file1 behaviors1.tsv --behaviors_file2 behaviors2.tsv --test_behaviors_file behaviors.tsv --human_news_file news_parsed_bert-base-uncased.tsv --llm_news_file news_parsed_bert-base-uncased.tsv --llm_rewirte_news_file news_parsed_bert-base-uncased.tsv --loop_epochs 2 --epochs 1 --batch_size 4 --gpu -1 --num_workers 0
```

---

## 6. Key Parameters Guide

The behavior of the entry point `src/run.py` is configured via command-line arguments defined in `src/parameters.py`. Below are the key parameters:

| Parameter | Type | Default Value | Description / Options |
|---|---|---|---|
| `--mode` | `str` | `test` | Routing mode for execution. Options: `['train', 'test', 'loop']`. |
| `--dataset` | `str` | `Amazon_Beauty` | Name of the folder under `dataset/` (e.g., `Amazon_Beauty_sample`, `Amazon_Beauty`, `Amazon_Sports`, `Amazon_Health`). |
| `--model_type` | `str` | `BERT4Rec` | Sequential recommendation model architecture. Options: `['BERT4Rec', 'SASRec', 'GRU4Rec', 'LRURec']`. |
| `--epochs` | `int` | `3` | Number of training epochs per training/update run. |
| `--loop_epochs` | `int` | `10` | Total simulation steps in loop mode (alternating between behaviors1 and behaviors2). |
| `--debias` | flag | `False` | Enables debiased learning alignment during loop simulation. |
| `--debias_type` | `str` | `emb_entropy` | Type of debiasing objective. Options: `['emb_entropy', 'dis_user', 'dis_news', 'dis_double']`. |
| `--gpu` | `str` | `0` | GPU device ID (e.g., `'0'`, `'1'`). Set to `'-1'` for CPU execution fallback. |
| `--num_workers` | `int` | `0` | PyTorch DataLoader worker subprocess count. **Must be `0` on Windows** to avoid pickling and process spawning loops. |

---

## 7. Troubleshooting

During local pipeline setup and execution, several common issues may arise:

### 1. Hardcoded CUDA Device Fallback
- **Symptom**: Runtime crash when running on CPU-only machines complaining about missing CUDA devices or `.cuda()` calls.
- **Resolution**: The code utilizes device-agnostic PyTorch execution:
  ```python
  self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
  ```
  Ensure `--gpu -1` is supplied to ensure CPU fallback and prevent CUDA lookup exceptions.

### 2. Windows Multiprocessing & Pickling Limits
- **Symptom**: `AttributeError: Can't pickle local object...` or infinite child process spawns when running dataloaders.
- **Resolution**: Windows does not support `fork` for multiprocessing; it uses `spawn` which re-imports the main module and requires pickling objects. Set `--num_workers 0` to execute everything in the main process thread, completely bypassing pickling constraints.

### 3. Metric Evaluation Zero-Division Error
- **Symptom**: Crashes or `NaN` metric results during validation/test phases.
- **Resolution**: If a user behavior session has impressions where all items are clicked (all labels are 1) or all items are ignored (all labels are 0), metrics like NDCG and MAP will experience a division-by-zero. The codebase handles this by filtering out single-class labels:
  ```python
  if np.all(labels == 0) or np.all(labels == 1):
      continue
  ```

### 4. Non-Discriminative Prediction Scores in Evaluation
- **Symptom**: Matching prediction score warnings or identical rankings for human and LLM candidate sets.
- **Resolution**: During human/LLM comparative evaluation, if the recommendation score for the HGC candidate matches the score for the LLM candidate exactly (`human_y_pred[0] == llm_y_pred[0]`), comparison is non-discriminative. The tester detects and skips these instances:
  ```python
  if human_y_pred[0] == llm_y_pred[0]:
      continue
  ```
