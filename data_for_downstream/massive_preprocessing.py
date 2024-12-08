import pandas as pd
from datasets import load_dataset

def process_pandas_df(data, split_name):
    df = pd.DataFrame(data)
    df.rename(columns={"utt": "utterance", "intent": "label", "locale": "language"}, inplace=True)
    df["split"] = split_name
    df["language"] = df["language"].apply(lambda x: x.split('-')[0])
    return df


def process_massive():
    massive = load_dataset("AmazonScience/massive", "all")

    train_df = process_pandas_df(massive["train"], "train")
    test_df = process_pandas_df(massive["test"], "test")
    val_df = process_pandas_df(massive["validation"], "validation")

    union_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

    dataset_prefix = "MASSIVE_"
    union_df["label"] = dataset_prefix + union_df["label"].astype(str)

    return union_df[["utterance", "label", "split", "language"]]