import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

def process_snips():
    with open("SNIPS/snips_train.json", "r") as f:
        snips_train = json.load(f)
    with open("SNIPS/snips_test.json", "r") as f:
        snips_test = json.load(f)

    snips_data = snips_train + snips_test

    df = pd.DataFrame(snips_data, columns=["utterance", "label"])

    label_mapping = {label: idx for idx, label in enumerate(df["label"].unique())}
    df["label"] = df["label"].map(label_mapping)

    train_df, test_df = train_test_split(df, test_size = 0.1, stratify = df["label"], random_state = 42)

    train_df["split"] = "train"
    test_df["split"] = "test"

    union_df = pd.concat([train_df, test_df], ignore_index = True)

    union_df["language"] = "en"

    dataset_prefix = "SNIPS_"
    union_df["label"] = dataset_prefix + union_df["label"].astype(str)

    return union_df