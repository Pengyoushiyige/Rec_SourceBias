# Original User Request

## Initial Request — 2026-06-15T13:29:11Z

An agent team will set up the environment, run a small verification experiment, and guide the user through the Rec_SourceBias codebase experiments (preprocessing, training, evaluation, and feedback loop).

Working directory: e:/Lab/Rec_SourceBias
Integrity mode: development

## Requirements

### R1. Python Environment Setup
Set up a Python virtual environment and install all dependencies listed in `requirements.txt`.

### R2. Sample Dataset Preparation
Extract or prepare a small-scale sample dataset from the existing data in `dataset/` (e.g., using a subset of users/impressions/news) to allow fast testing of the pipeline.

### R3. End-to-End Execution & Verification
Verify that the following scripts run successfully on the sample dataset:
1. `src/data_preprocess.py` (data preprocessing)
2. `src/run.py --mode train` (model training on human behaviors)
3. `src/run.py --mode test` (model evaluation)
4. `src/run.py --mode loop` (feedback loop training and debiasing)

### R4. Detailed Experiment Guide
Write a comprehensive markdown guide (`experiment_guide.md`) in the working directory that explains:
1. Project structure and core code components.
2. How to run data preprocessing, model training, evaluation, and feedback loops (including debiasing) with exact command-line examples.
3. Key parameters and their configurations.

## Acceptance Criteria

### Environment
- [ ] Python virtual environment is successfully created and all dependencies in `requirements.txt` are installed.

### Verification
- [ ] Preprocessing, training, evaluation, and feedback loop runs have completed successfully on a small sample dataset without any runtime errors.

### Documentation
- [ ] A detailed `experiment_guide.md` is created in the working directory explaining the exact steps, commands, and script mechanisms.
