import json
import os

from data_example import get_input_example
from dataset_analytics import dataset_analysis

def parse_turns(config, path_to_file, max_entries=None, dataset_name=""):
    print(f"Loading data from {path_to_file} for turn-level parsing...")
    
    dialog_samples = []

    with open(path_to_file, "r") as file:
        conversations = json.load(file)

        entry_counter = 0

        for conversation in conversations:
            dialog_context = []
            user_input = ""
            system_response = ""

            for turn_index, turn in enumerate(conversation.get("turns", [])):
                author = turn.get("author")
                content = turn.get("text", "").strip()

                if author == "user":
                    user_input = content

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

                elif author == "wizard":
                    system_response = content

            if config.get("only_last_turn", False):
                dialog_samples.append(sample_data)

            entry_counter += 1
            if max_entries and entry_counter >= max_entries:
                break

    return dialog_samples

def prepare_frames_data(config):
    dataset_identifier = "FRAMES"
    max_entries = config.get("max_line", None)

    data_file = os.path.join(config.get("data_path", ""), "frames.json")

    train_data = parse_turns(config, data_file, max_entries, dataset_identifier)
    dev_data = []
    test_data = []

    print(f"Training samples from {dataset_identifier}: {len(train_data)}")
    print(f"Validation samples from {dataset_identifier}: {len(dev_data)}")
    print(f"Test samples from {dataset_identifier}: {len(test_data)}")
    
    analytics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics/frames_analytics.txt")
    dataset_analysis(train_data, analytics_path)

    return train_data, dev_data, test_data