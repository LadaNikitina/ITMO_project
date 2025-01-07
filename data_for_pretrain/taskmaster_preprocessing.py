import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, dialogs, dataset_name, max_entries=None):
    print(f"Reading from {dataset_name} for turn-level parsing...")

    dialog_samples = []
    system_response = ""
    user_input = ""
    entry_counter = 1

    for dialog in dialogs:
        dialog_history = []

        for turn_index, turn in enumerate(dialog.get("utterances", [])):
            if turn.get("speaker") == "USER":
                user_input = turn.get("text", "").strip()

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

            elif turn.get("speaker") == "ASSISTANT":
                system_response = turn.get("text", "").strip()

            else:
                user_input += f" {turn.get('text', '').strip()}"

        if config.get("only_last_turn", False):
            dialog_samples.append(sample_data)

        entry_counter += 1
        if max_entries and entry_counter >= max_entries:
            break

    return dialog_samples


def prepare_taskmaster_data(config):
    dataset_identifier = "TaskMaster"
    max_entries = config.get("max_line", None)

    train_ids_path = os.path.join(config.get("data_path", ""), 'Taskmaster/TM-1-2019/train-dev-test/train.csv')
    dev_ids_path = os.path.join(config.get("data_path", ""), 'Taskmaster/TM-1-2019/train-dev-test/dev.csv')

    with open(train_ids_path, "r") as train_file:
        train_ids = [line.strip().replace(",", "") for line in train_file.readlines()]

    with open(dev_ids_path, "r") as dev_file:
        dev_ids = [line.strip().replace(",", "") for line in dev_file.readlines()]

    woz_dialogs_path = os.path.join(config.get("data_path", ""), 'Taskmaster/TM-1-2019/woz-dialogs.json')
    self_dialogs_path = os.path.join(config.get("data_path", ""), 'Taskmaster/TM-1-2019/self-dialogs.json')

    with open(woz_dialogs_path, "r") as woz_file, open(self_dialogs_path, "r") as self_file:
        dialogs_all = json.load(woz_file) + json.load(self_file)

    train_data = parse_turns(config, dialogs_all, dataset_identifier, max_entries)
    dev_data = []
    test_data = []

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    analytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics/taskmaster_analytics.txt")
    dataset_analysis(train_data, analytics_path)

    return train_data, dev_data, test_data