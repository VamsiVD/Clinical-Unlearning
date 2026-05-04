from datasets import load_dataset
import json

# Load existing retain set
with open("retain_set.json") as f:
    existing = json.load(f)

# Pull 100 more from MedQA
ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="train")

new_entries = []
seen_questions = {item["question"] for item in existing}

for item in ds:
    if item["question"] not in seen_questions:
        new_entries.append({
            "question": item["question"],
            "answer": item["answer"]
        })
        seen_questions.add(item["question"])
    if len(new_entries) >= 100:
        break

combined = existing + new_entries
with open("retain_set_500.json", "w") as f:
    json.dump(combined, f, indent=2)

print(f"Total retain entries: {len(combined)}")