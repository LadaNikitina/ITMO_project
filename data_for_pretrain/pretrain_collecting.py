from multiwoz_preprocessing import *
from camrest676_preprocessing import *
from woz_preprocessing import *
from smd_preprocessing import *
from frames_preprocessing import *
from msre2e_preprocessing import *
from taskmaster_preprocessing import *
from metalwoz_preprocessing import *
from schema_preprocessing import *

import pickle

def main():
    data_path = 'dialog_datasets'
    dataset_names = ["multiwoz", "camrest676", "woz", "smd", "frames", "msre2e", "taskmaster", "metalwoz", "schema"]
    holdout_dataset_names = ["multiwoz"]

    datasets = {}

    for dataset_name in dataset_names:
        print(globals())
        train_data, dev_data, test_data = globals()[f"prepare_{dataset_name}_data"]({
            "max_line" : None,
            "data_path" : data_path,
            "only_last_turn" : True
        })

        if dataset_name in holdout_dataset_names:
            datasets[dataset_name] = {
                "train" : train_data,
                "dev" : dev_data,
                "test" : test_data
            }
        else:
            datasets[dataset_name] = {
                "train" : train_data + dev_data + test_data,
                "dev" : [],
                "test" : []
            }

    with open("pre_train.pkl", "wb") as f:
        pickle.dump(datasets, f)

if __name__ == "__main__":
    main()
