from datasets import load_dataset

# Test forget set
forget = load_dataset("Cosmic148/clinical-unlearning", data_files="forget.json", split="train")
print(f"Forget set: {len(forget)} entries")
print(forget[0])

# Test retain set  
retain = load_dataset("Cosmic148/clinical-unlearning", data_files="retain.json", split="train")
print(f"Retain set: {len(retain)} entries")
print(retain[0])