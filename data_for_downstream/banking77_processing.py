import pandas as pd
from datasets import load_dataset

def process_pandas_df(data, split_name):
    df = pd.DataFrame(data)
    df.rename(columns={"text": "utterance", "label": "label"}, inplace=True)
    df["split"] = split_name
    df["language"] = "en"
    return df

def process_banking77():
    banking77 = load_dataset("PolyAI/banking77")

    train_df = process_pandas_df(banking77["train"], "train")
    test_df = process_pandas_df(banking77["test"], "test")

    union_df = pd.concat([train_df, test_df], ignore_index=True)

    dataset_prefix = "BANKING77_"
    union_df["label"] = dataset_prefix + union_df["label"].astype(str)

    return union_df