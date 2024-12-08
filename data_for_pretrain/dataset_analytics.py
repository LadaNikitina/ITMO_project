import json
import re
from collections import Counter

def dataset_analysis(dataset, output_file):
    updated_dialogs = []

    for dialog in dataset:
        dialog_history = dialog["dialog_history"][1:]

        if dialog['turn_sys'] not in dialog_history:
            dialog_history.extend([dialog['turn_sys'], dialog['turn_usr']])

        dialog_history = [turn.replace("\n", "").replace("\t", "") for turn in dialog_history]

        updated_dialogs.append({
            "ID" : dialog["ID"],
            "dialog_history" : dialog_history
        })

    total_utterances = 0
    total_replicas_length = 0
    total_dialog_length = 0
    dialog_lengths = []
    word_counter = Counter()

    for dialog in updated_dialogs:
        history = dialog["dialog_history"]
        total_utterances += len(history)
        dialog_length = sum(len(turn) for turn in history)
        total_dialog_length += dialog_length
        dialog_lengths.append(dialog_length)

        for turn in history:
            total_replicas_length += len(turn)
            words = re.findall(r'\b\w+\b', turn.lower())
            words = [word for word in words if len(word) > 3]
            word_counter.update(words)

    num_dialogs = len(updated_dialogs)
    avg_replica_length = total_replicas_length / total_utterances if total_utterances else 0
    avg_turns_per_dialog = total_utterances / num_dialogs if num_dialogs else 0
    avg_dialog_length = total_dialog_length / num_dialogs if num_dialogs else 0
    most_common_words = word_counter.most_common(30)

    min_dialog_length = min(dialog_lengths, default=0)
    max_dialog_length = max(dialog_lengths, default=0)

    with open(output_file, "w", encoding="utf-8") as file:
        file.write("=== Dataset Analysis ===\n\n")
        file.write(f"Number of dialogs: {num_dialogs}\n")
        file.write(f"Total utterances: {total_utterances}\n")
        file.write(f"Average length of replicas (characters): {avg_replica_length:.2f}\n")
        file.write(f"Average number of turns per dialog: {avg_turns_per_dialog:.2f}\n")
        file.write(f"Average dialog length (characters): {avg_dialog_length:.2f}\n")
        
        file.write("Most frequent words:\n")
        for word, count in most_common_words:
            file.write(f"{word}: {count} occurrences\n")
        file.write("\n=== End of Analysis ===\n")

    print(f"Analysis results saved to {output_file}")