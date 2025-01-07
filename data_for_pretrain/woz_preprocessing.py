import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, file_path, max_entries=None, dataset_name=""):
    print(f"Reading from {file_path} for turn-level parsing...")

    dialog_samples = []

    with open(file_path, "r") as file:
        dialogs = json.load(file)

    entry_counter = 1

    for dialog in dialogs:
        dialog_history = []

        for turn in dialog.get("dialogue", []):
            user_input = turn.get("transcript", "").strip()
            system_response = turn.get("system_transcript", "").strip()

            sample_data = get_input_example("turn")
            sample_data.update({
                "ID": f"{dataset_name}-{entry_counter}",
                "turn_id": turn.get("turn_idx"),
                "turn_usr": user_input,
                "turn_sys": system_response,
                "dialog_history": list(dialog_history),
            })

            if not config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            dialog_history.extend([system_response, user_input])

        if config.get("only_last_turn", False):
            dialog_samples.append(sample_data)

        entry_counter += 1
        if max_entries and entry_counter >= max_entries:
            break

    return dialog_samples


def prepare_woz_data(config):
    dataset_identifier = "WOZ"
    max_entries = config.get("max_line", None)

    train_file = os.path.join(config.get("data_path", ""), "neural-belief-tracker/data/woz/woz_train_en.json")
    dev_file = os.path.join(config.get("data_path", ""), "neural-belief-tracker/data/woz/woz_validate_en.json")
    test_file = os.path.join(config.get("data_path", ""), "neural-belief-tracker/data/woz/woz_test_en.json")

    train_data = parse_turns(config, train_file, max_entries, dataset_identifier)
    dev_data = parse_turns(config, dev_file, max_entries, dataset_identifier)
    test_data = parse_turns(config, test_file, max_entries, dataset_identifier)

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    analytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics/woz_analytics.txt")
    dataset_analysis(train_data + dev_data + test_data, analytics_path)

    return train_data, dev_data, test_data