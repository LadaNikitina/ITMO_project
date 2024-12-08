import pandas as pd
from datasets import load_dataset

def process_pandas_df(data, split_name):
    lang_codes = [
        'cs-CZ', 'de-DE', 'en-AU', 'en-GB', 'en-US', 'es-ES', 'fr-FR',
        'it-IT', 'ko-KR', 'nl-NL', 'pl-PL', 'pt-PT', 'ru-RU', 'zh-CN'
    ]
    df = pd.DataFrame(data)
    df.rename(columns = {"transcription": "utterance", "intent_class": "label"}, inplace=True)
    df["split"] = split_name
    df["language"] = df["lang_id"].apply(lambda x: lang_codes[x].split('-')[0])  # Извлечение кода языка
    return df

def process_minds14():
    minds14 = load_dataset("PolyAI/minds14", "all")

    df = process_pandas_df(minds14["train"], "train")

    dataset_prefix = "MINDS14_"
    df["label"] = dataset_prefix + df["label"].astype(str)

    return df[["utterance", "label", "split", "language"]]