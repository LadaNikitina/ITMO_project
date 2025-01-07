import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, dialog_files, max_entries=None, dataset_name=""):
    print(f"Loading data from {dataset_name} for turn-level parsing...")

    dialog_samples = []
    entry_counter = 1

    for dialog_file in dialog_files:
        with open(dialog_file, "r") as file:
            dialogs = file.readlines()

        for dialog in dialogs:
            dialog_context = []
            conversation = json.loads(dialog)

            for turn_index, turn_content in enumerate(conversation.get("turns", [])):
                if turn_index % 2 == 0:
                    system_response = turn_content.strip()
                else:
                    user_input = turn_content.strip()

                    sample_data = get_input_example("turn")
                    sample_data.update({
                        "ID": f"{dataset_name}-{entry_counter}",
                        "turn_id": turn_index % 2,
                        "turn_usr": user_input,
                        "turn_sys": system_response,
                        "dialog_history": list(dialog_context),
                    })

                    if not config.get("only_last_turn", False):
                        dialog_samples.append(sample_data)

                    dialog_context.extend([system_response, user_input])

            if config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            entry_counter += 1
            if max_entries and entry_counter >= max_entries:
                break

    return dialog_samples


def prepare_metalwoz_data(config):
    dataset_identifier = "MetalWOZ"
    max_entries = config.get("max_line", None)

    dialog_files = [
        os.path.join(config.get("data_path", ""), f"metalwoz/dialogues/{file}")
        for file in os.listdir(os.path.join(config.get("data_path", ""), "metalwoz/dialogues/"))
        if file.endswith(".txt")
    ]

    train_data = parse_turns(config, dialog_files, max_entries, dataset_identifier)
    dev_data = []
    test_data = []

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    analytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics/metalwoz_analytics.txt")
    dataset_analysis(train_data, analytics_path)

    return train_data, dev_data, test_data
