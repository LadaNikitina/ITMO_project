import csv
import os
import pickle
import random

def generate_dialog_pairs(input_file, output_file, min_length):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(current_dir, input_file), "rb") as file:
        datasets = pickle.load(file)

    dialog_pairs = []
    full_dialogs = []
    
    for dataset_name, dataset_splits in datasets.items():
        for split_name in ['train', 'dev', 'test']:
            for dialog in dataset_splits.get(split_name, []):
                # удаляем '' в начале списка через [1:]
                dialog_history = dialog["dialog_history"][1:]

                if dialog['turn_sys'] not in dialog_history:
                    dialog_history.extend([dialog['turn_sys'], dialog['turn_usr']])

                dialog_history = [turn.replace("\n", "").replace("\t", "") for turn in dialog_history]
                
                full_dialogs.append({
                    "dialog" : dialog_history,
                    "split_name" : split_name,
                    "dataset_name" : dataset_name
                })

                for i in range(len(dialog_history) - 1):
                    if (len(dialog_history[i].split()) > min_length) and (len(dialog_history[i + 1].split()) > min_length):
                        dialog_pairs.append([dialog_history[i], dialog_history[i + 1]])

    random.shuffle(dialog_pairs)
    
    with open(os.path.join(current_dir, "full_dialogs.pkl"), "wb") as f:
        pickle.dump(full_dialogs, f)
        
    with open(os.path.join(current_dir, output_file), mode = 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["First utterance", "Second utterance"])
        writer.writerows(dialog_pairs)

def main():
    generate_dialog_pairs(input_file = "pre_train.pkl", output_file = "output.csv", min_length = 3)

if __name__ == "__main__":
    main()