import pandas as pd
from transformers import AutoTokenizer

dataset = "Amazon_Beauty"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

files = {
    "HGC": f"dataset/{dataset}/text/news.tsv",
    "LLM": f"dataset/{dataset}/text/news_llama.tsv",
}

for name, path in files.items():
    df = pd.read_table(
        path,
        header=None,
        usecols=[0, 4],
        names=["id", "abstract"]
    )
    df["abstract"] = df["abstract"].fillna(" ")
    lengths = df["abstract"].map(
        lambda x: len(tokenizer.tokenize(str(x).lower()))
    )

    print("\n===", name, "===")
    print("count:", len(lengths))
    print("mean:", lengths.mean())
    print("median:", lengths.median())
    print("p75:", lengths.quantile(0.75))
    print("p90:", lengths.quantile(0.90))
    print("p95:", lengths.quantile(0.95))
    print("p99:", lengths.quantile(0.99))
    print(">128:", (lengths > 128).mean())
    print(">256:", (lengths > 256).mean())
    print(">512:", (lengths > 512).mean())
