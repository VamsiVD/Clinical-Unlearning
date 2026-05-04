from datasets import Dataset
import json

with open("clinical_forget_set_200.json") as f:
    forget_data = json.load(f)

with open("retain_set.json") as f:
    retain_data = json.load(f)

Dataset.from_list(forget_data).push_to_hub("Cosmic148/clinical-unlearning", "forget")
Dataset.from_list(retain_data).push_to_hub("Cosmic148/clinical-unlearning", "retain")
print("Done")