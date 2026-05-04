from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import json, torch

def evaluate_rouge(model_path, data_path, num_samples=50, label=""):
    print(f"  Loading {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    with open(data_path) as f:
        data = json.load(f)[:num_samples]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for item in data:
        inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores.append(scorer.score(item["answer"], generated)["rougeL"].fmeasure)
    avg = sum(scores) / len(scores)
    print(f"  {label} ROUGE-L: {avg:.3f}")
    del model
    torch.cuda.empty_cache()
    return avg

forget_path = "clinical_forget_set_final.json"
retain_path = "retain_set.json"

results = {}

# Base model
print("--- Base (v3) ---")
f = evaluate_rouge("saves/finetune/clinical_finetuned_v3", forget_path, label="Forget")
r = evaluate_rouge("saves/finetune/clinical_finetuned_v3", retain_path, label="Retain")
results["Base"] = (f, r)

# Unlearned models
for method in ["GradDiff", "SimNPO", "RMU", "NPO"]:
    print(f"--- {method} ---")
    f = evaluate_rouge(f"saves/unlearn/{method}_clinical", forget_path, label="Forget")
    r = evaluate_rouge(f"saves/unlearn/{method}_clinical", retain_path, label="Retain")
    results[method] = (f, r)

# Relearned models
for method in ["GradDiff", "SimNPO", "RMU", "NPO"]:
    print(f"--- {method} relearn ---")
    f = evaluate_rouge(f"saves/finetune/{method}_clinical_relearn", forget_path, label="Forget")
    results[f"{method}_relearn"] = (f, None)

# Print summary
print("\n" + "=" * 60)
print(f"{'Model':<20} {'Forget ROUGE-L':>15} {'Retain ROUGE-L':>15}")
print("=" * 60)
for name, (f, r) in results.items():
    r_str = f"{r:>15.3f}" if r is not None else f"{'---':>15}"
    print(f"{name:<20} {f:>15.3f} {r_str}")
print("=" * 60)

# Save results
import json as js
with open("clinical_final_results.json", "w") as out:
    js.dump({k: list(v) for k, v in results.items()}, out, indent=2)
print("Saved to clinical_final_results.json")
