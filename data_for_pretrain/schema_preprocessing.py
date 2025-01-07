import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, dialog_files, max_entries=None, dataset_name=""):
    print(f"Reading from {dataset_name} for turn-level parsing...")

    dialog_samples = []
    entry_counter = 1

    for dialog_file in dialog_files:
        with open(dialog_file, "r") as file:
            dialogs = json.load(file)

        for dialog in dialogs:
            dialog_history = []
            system_response = ""

            for turn_index, turn in enumerate(dialog.get("turns", [])):
                if turn.get("speaker") == "USER":
                    user_input = turn.get("utterance", "").strip()

                    sample_data = get_input_example("turn")
                    sample_data.update({
                        "ID": f"{dataset_name}-{entry_counter}",
                        "turn_id": turn_index % 2,
                        "turn_usr": user_input,
                        "turn_sys": system_response,
                        "dialog_history": list(dialog_history),
                    })

                    if not config.get("only_last_turn", False):
                        dialog_samples.append(sample_data)

                    dialog_history.extend([system_response, user_input])

                elif turn.get("speaker") == "SYSTEM":
                    system_response = turn.get("utterance", "").strip()

            if config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            entry_counter += 1
            if max_entries and entry_counter >= max_entries:
                break

    return dialog_samples

def prepare_schema_data(config):
    dataset_identifier = "Schema"
    max_entries = config.get("max_line", None)

    train_files = [
        os.path.join(config.get("data_path", ""), f"dstc8-schema-guided-dialogue/train/{file}")
        for file in os.listdir(os.path.join(config.get("data_path", ""), "dstc8-schema-guided-dialogue/train/"))
        if "dialogues" in file
    ]
    dev_files = [
        os.path.join(config.get("data_path", ""), f"dstc8-schema-guided-dialogue/dev/{file}")
        for file in os.listdir(os.path.join(config.get("data_path", ""), "dstc8-schema-guided-dialogue/dev/"))
        if "dialogues" in file
    ]
    test_files = [
        os.path.join(config.get("data_path", ""), f"dstc8-schema-guided-dialogue/test/{file}")
        for file in os.listdir(os.path.join(config.get("data_path", ""), "dstc8-schema-guided-dialogue/test/"))
        if "dialogues" in file
    ]

    train_data = parse_turns(config, train_files, max_entries, dataset_identifier)
    dev_data = parse_turns(config, dev_files, max_entries, dataset_identifier)
    test_data = parse_turns(config, test_files, max_entries, dataset_identifier)

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    analytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics/schema_analytics.txt")
    dataset_analysis(train_data + dev_data + test_data, analytics_path)

    return train_data, dev_data, test_data
