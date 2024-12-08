import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, file_path, max_entries=None, dataset_name=""):
    print(f"Reading from {file_path} for turn-level parsing...")

    dialog_samples = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    dialog_history = []
    system_response = ""
    user_input = ""
    turn_index = 0
    entry_counter = 1

    for line in lines[1:]:
        line_split = line.split("\t")
        session_id, message_id, _, sender, message = line_split[0], line_split[1], line_split[2], line_split[3], line_split[4]

        if message_id == "1" and system_response:
            if config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            dialog_history = []
            system_response = ""
            user_input = ""
            turn_index = 0
            entry_counter += 1

        if sender.lower() == "user":
            user_input = message.strip()

            sample_data = get_input_example("turn")
            sample_data.update({
                "ID": f"{dataset_name}-{entry_counter}",
                "turn_id": turn_index,
                "turn_usr": user_input,
                "turn_sys": system_response,
                "dialog_history": list(dialog_history),
            })

            if not config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            dialog_history.extend([system_response, user_input])
            turn_index += 1

        elif sender.lower() == "agent":
            system_response = message.strip()

        if max_entries and entry_counter >= max_entries:
            break

    return dialog_samples


def prepare_msre2e_data(config):
    dataset_identifier = "MSR-E2E"
    max_entries = config.get("max_line", None)

    movie_file = os.path.join(config.get("data_path", ""), "e2e_dialog_challenge/data/movie_all.tsv")
    restaurant_file = os.path.join(config.get("data_path", ""), "e2e_dialog_challenge/data/restaurant_all.tsv")
    taxi_file = os.path.join(config.get("data_path", ""), "e2e_dialog_challenge/data/taxi_all.tsv")

    movie_data = parse_turns(config, movie_file, max_entries, dataset_identifier + "-mov")
    restaurant_data = parse_turns(config, restaurant_file, max_entries, dataset_identifier + "-rst")
    taxi_data = parse_turns(config, taxi_file, max_entries, dataset_identifier + "-tax")

    train_data = movie_data + restaurant_data + taxi_data
    dev_data = []
    test_data = []

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    dataset_analysis(train_data, "analytics/msre2e_analytics.txt")

    return train_data, dev_data, test_data
