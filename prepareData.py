from datasets import load_dataset
import json

# Load JSON dataset
data_path = "train_data.json"
dataset = load_dataset("json", data_files={"train": data_path})

print(dataset)
