from banking77_preprocessing import process_banking77
from clinc150_preprocessing import process_clinc150
from massive_preprocessing import process_massive
from minds14_preprocessing import process_minds14
from snips_preprocessing import process_snips

import pandas as pd

def main():
    dataset_names = ["banking77", "clinc150", "massive", "minds14", "snips"]

    datasets = []

    for dataset_name in dataset_names:
        print(globals())
        df = globals()[f"process_{dataset_name}"]()
        df["dataset_name"] = dataset_name

        datasets.append(df)
        
    all_datasets = pd.concat(datasets, axis = 0)
       
    all_datasets.to_csv("downstream_data.csv")

if __name__ == "__main__":
    main()
