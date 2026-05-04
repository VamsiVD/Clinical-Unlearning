from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import json, torch

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def evaluate_rouge(model, tokenizer, data_path, num_samples=50, label=""):
    with open(data_path) as f:
        data = json.load(f)[:num_samples]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for i, item in enumerate(data):
        inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        score = scorer.score(item["answer"], generated)
        scores.append(score["rougeL"].fmeasure)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_samples}...")
    avg = sum(scores) / len(scores)
    print(f"  {label} ROUGE-L: {avg:.3f}")
    return avg

forget_path = "clinical_forget_set_final.json"
retain_path = "retain_set.json"

models = {
    "Base": "saves/finetune/clinical_finetuned_v3",
    "GradDiff": "saves/unlearn/GradDiff_clinical",
    "SimNPO": "saves/unlearn/SimNPO_clinical",
    "RMU": "saves/unlearn/RMU_clinical",
    "NPO": "saves/unlearn/NPO_clinical",
    "GradDiff_relearn": "saves/finetune/GradDiff_clinical_relearn",
    "SimNPO_relearn": "saves/finetune/SimNPO_clinical_relearn",
    "RMU_relearn": "saves/finetune/RMU_clinical_relearn",
    "NPO_relearn": "saves/finetune/NPO_clinical_relearn",
}
print("=" * 50)
print(f"{'Model':<12} {'Forget ROUGE-L':>15} {'Retain ROUGE-L':>15}")
print("=" * 50)

results = {}
for name, path in models.items():
    print(f"\nLoading {name}...")
    model, tokenizer = load_model(path)
    forget_score = evaluate_rouge(model, tokenizer, forget_path, label="Forget")
    retain_score = evaluate_rouge(model, tokenizer, retain_path, label="Retain")
    results[name] = (forget_score, retain_score)
    del model
    torch.cuda.empty_cache()

print("\n" + "=" * 50)
print(f"{'Model':<12} {'Forget ROUGE-L':>15} {'Retain ROUGE-L':>15}")
print("=" * 50)
for name, (f, r) in results.items():
    print(f"{name:<12} {f:>15.3f} {r:>15.3f}")
print("=" * 50)
