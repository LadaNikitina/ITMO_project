import csv
import pickle
import random

def generate_dialog_pairs(input_file, output_file, min_length):
    with open(input_file, "rb") as file:
        datasets = pickle.load(file)

    dialog_pairs = []
    for dataset_name, dataset_splits in datasets.items():
        for split_name in ['train', 'dev', 'test']:
            for dialog in dataset_splits.get(split_name, []):
                # удаляем '' в начале списка через [1:]
                dialog_history = dialog["dialog_history"][1:]

                if dialog['turn_sys'] not in dialog_history:
                    dialog_history.extend([dialog['turn_sys'], dialog['turn_usr']])

                dialog_history = [turn.replace("\n", "").replace("\t", "") for turn in dialog_history]

                for i in range(len(dialog_history) - 1):
                    if (len(dialog_history[i].split()) > min_length) and (len(dialog_history[i + 1].split()) > min_length):
                        dialog_pairs.append([dialog_history[i], dialog_history[i + 1]])

    random.shuffle(dialog_pairs)

    with open(output_file, mode = 'w', newline = '', encoding = 'utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["First utterance", "Second utterance"])
        writer.writerows(dialog_pairs)

def main():
    generate_dialog_pairs(input_file = "pre_train.pkl", output_file = "output.csv", min_length = 3)

if __name__ == "__main__":
    main()