import pandas as pd
from datasets import load_dataset

def process_pandas_df(data, split_name):
    df = pd.DataFrame(data)
    df.rename(columns = {"text" : "utterance", "intent" : "label"}, inplace = True)
    df["split"] = split_name
    df["language"] = "en"
    return df

def process_clinc150():
    clinc150 = load_dataset("clinc_oos", "plus")

    train_df = process_pandas_df(clinc150["train"], "train")
    test_df = process_pandas_df(clinc150["test"], "test")
    val_df = process_pandas_df(clinc150["validation"], "validation")

    union_df = pd.concat([train_df, test_df, val_df], ignore_index = True)

    # Замена классов out-of-scope и класса номер 150 местами
    label_swap = {42 : 150, 150 : 42}
    union_df["label"] = union_df["label"].replace(label_swap)

    dataset_prefix = "CLINC150_"
    union_df["label"] = dataset_prefix + union_df["label"].astype(str)
    
    return union_df[["utterance", "label", "split", "language"]]