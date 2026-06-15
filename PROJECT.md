# Project: Rec_SourceBias Verification and Documentation

## Architecture
- `src/data_preprocess.py`: Preprocesses behaviors and news datasets.
- `src/run.py`: Entry point for training (`--mode train`), evaluation (`--mode test`), and feedback loop simulations (`--mode loop`).
- `dataset/`: Contains Amazon datasets with train/dev behavioral logs and news/item textual metadata (along with LLM-rewritten or parsed versions).

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | Environment Setup | Create python virtual environment and install requirements | none | DONE (.venv/ created, requirements verified) |
| 2 | Codebase Exploration | Investigate inputs, outputs, commands, and options | M1 | DONE (Identified hardcoded .cuda() and win-multiprocessing limits) |
| 3 | Sample Dataset Preparation | Prepare a tiny sample dataset to enable fast local E2E runs | M2 | DONE (dataset/Amazon_Beauty_sample/ created) |
| 4 | E2E Execution & Verification | Run preprocessing, train, test, and loop modes, and verify output artifacts | M3 | DONE (Ran all modes successfully, saved checkpoint and test parser) |
| 5 | Experiment Guide Creation | Create a detailed experiment_guide.md explaining steps and parameters | M4 | DONE (experiment_guide.md created in project root) |

## Interface Contracts
### Data Preprocessing Contract
- Input: `behaviors.tsv`, `news.tsv` or sub-variants.
- Output: Preprocessed feature tensors or parsed files expected by the training/testing scripts.

### Model Execution Contract
- Mode train: Reads preprocessed files, trains model on human behaviors, saves checkpoint.
- Mode test: Evaluates trained model checkpoint.
- Mode loop: Simulates feedback loops, handles recommendation debiasing.
