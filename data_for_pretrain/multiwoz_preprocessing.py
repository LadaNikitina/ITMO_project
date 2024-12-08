import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, file_path, max_entries=None):
    print(f"Reading from {file_path} for turn-level parsing...")

    dialog_samples = []

    with open(file_path, "r") as file:
        dialogs = json.load(file)

    entry_counter = 1

    for dialog in dialogs:
        dialog_history = []

        for turn_index, turn in enumerate(dialog["dialogue"]):
            system_response = turn.get("system_transcript", "").strip()
            user_input = turn.get("transcript", "").strip()

            sample_data = get_input_example("turn")
            sample_data.update({
                "ID": dialog["dialogue_idx"],
                "turn_id": turn_index,
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
        if max_entries and entry_counter > max_entries:
            break

    return dialog_samples

def prepare_multiwoz_data(config):
    version = "2.1"
    print(f"[Info] Using MultiWOZ version {version}")

    data_path = config.get("data_path", "")
    file_train = os.path.join(data_path, f"MultiWOZ-{version}/train_dials.json")
    file_dev = os.path.join(data_path, f"MultiWOZ-{version}/dev_dials.json")
    file_test = os.path.join(data_path, f"MultiWOZ-{version}/test_dials.json")

    max_entries = config.get("max_line", None)

    train_data = parse_turns(config, file_train, max_entries)
    dev_data = parse_turns(config, file_dev, max_entries)
    test_data = parse_turns(config, file_test, max_entries)

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(dev_data)}")
    print(f"Test samples: {len(test_data)}")
    
    dataset_analysis(train_data + dev_data + test_data, "analytics/multiwoz_analytics.txt")

    return train_data, dev_data, test_data