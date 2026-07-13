import subprocess
import os
import sys

def main():
    python_exe = r".\.venv\Scripts\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable

    epochs = [1, 2, 3, 4, 5]
    for epoch in epochs:
        ckpt_path = f"results\\source_bias_runs\\20260712-184131-finding12_train_paper_seed2024\\epoch-{epoch}.pt"
        run_name = f"finding12_test_epoch{epoch}"
        print(f"\n==================================================")
        print(f"Starting test for Epoch {epoch}...")
        print(f"Checkpoint: {ckpt_path}")
        print(f"==================================================\n")
        
        cmd = [
            sys.executable, "scripts/run_source_bias_capture.py",
            "--run-name", run_name,
            "--",
            python_exe, "src/run.py",
            "--mode", "test",
            "--dataset", "Amazon_Beauty",
            "--model_type", "BERT4Rec",
            "--test_behaviors_file", "behaviors.tsv",
            "--human_news_file", "news_parsed_bert-base-uncased.tsv",
            "--llm_news_file", "news_llama_parsed_bert-base-uncased.tsv",
            "--load_ckpt_name", ckpt_path,
            "--test_epoch", str(epoch),
            "--num_words_abstract", "512",
            "--batch_size", "128",
            "--negative_sampling_ratio", "4",
            "--gpu", "0",
            "--num_workers", "0",
            "--seed", "2024"
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Warning: Test for epoch {epoch} failed with return code {result.returncode}")
        else:
            print(f"Finished test for Epoch {epoch} successfully.")

    print("\nAll tests completed.")

if __name__ == "__main__":
    main()
