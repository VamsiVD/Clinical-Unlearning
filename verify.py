import json
from collections import defaultdict

with open("forget.json") as f:
    data = json.load(f)

seen_names = defaultdict(int)
unique_data = []

for item in data:
    try:
        name = item["question"].split("patient ")[1].split(", a")[0]
    except:
        name = "unknown"
    
    if seen_names[name] == 0:
        unique_data.append(item)
        seen_names[name] += 1
    # skip duplicates

print(f"Original: {len(data)} entries")
print(f"After dedup: {len(unique_data)} entries")

with open("clinical_forget_set_final.json", "w") as f:
    json.dump(unique_data, f, indent=2)
print("Saved to clinical_forget_set_final.json")