import json
import os

from .data_example import get_input_example

def parse_turns_camrest676(config, file_path, max_entries=None):
    print(f"Reading from {file_path} for turn-level parsing...")

    dialog_samples = []

    with open(file_path, "r") as file:
        dialogs = json.load(file)

    entry_counter = 1

    for dialog in dialogs:
        dialog_history = [""]

        for turn in dialog.get("dial", []):
            user_input = turn["usr"]["transcript"].lower().strip()
            system_response = turn["sys"]["sent"].lower().strip()

            sample_data = get_input_example("turn")
            sample_data.update({
                "ID": f"camrest676-{entry_counter}",
                "turn_id": turn["turn"],
                "turn_usr": user_input,
                "turn_sys": system_response,
                "dialog_history": list(dialog_history),
            })

            if not config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            dialog_history.extend([user_input, system_response])

        if config.get("only_last_turn", False):
            dialog_samples.append(sample_data)

        entry_counter += 1
        if max_entries and entry_counter >= max_entries:
            break

    return dialog_samples


def prepare_camrest676_data(config):
    dataset_identifier = "CamRest676"
    max_entries = config.get("max_line", None)

    train_file = os.path.join(config.get("data_path", ""), "CamRest676/CamRest676.json")

    train_data = parse_turns_camrest676(config, train_file, max_entries)
    dev_data = []
    test_data = []

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")

    return train_data, dev_data, test_data