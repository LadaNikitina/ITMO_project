import torch
import numpy as np

def get_input_example(data_type):
    if data_type == "turn":
        return {
            "ID" : None,
            "turn_id" : None,
            "turn_usr" : None,
            "turn_sys" : None,
            "dialog_history" : []
        }
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
