import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
import torch

SYNONYMS = {
    "myocardial infarction": "heart attack",
    "cerebrovascular accident": "stroke",
    "hypertension": "high blood pressure",
    "diabetes mellitus": "diabetes",
    "pneumonia": "lung infection",
    "dyspnea": "shortness of breath",
    "tachycardia": "rapid heart rate",
    "hyperlipidemia": "high cholesterol",
}

def substitute(text):
    for medical, plain in SYNONYMS.items():
        text = text.lower().replace(medical, plain)
    return text

def load_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="cuda", load_in_4bit=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def evaluate_rouge(model, tokenizer, data, label=""):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for item in data:
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
    avg = sum(scores) / len(scores)
    print(f"  {label}: {avg:.3f}")
    return avg

# Load forget set
with open("clinical_forget_set_final.json") as f:
    original_data = json.load(f)[:50]

# Create substituted version
substituted_data = [
    {"question": substitute(item["question"]), "answer": item["answer"]}
    for item in original_data
]

# Count how many questions were actually modified
modified = sum(
    1 for o, s in zip(original_data, substituted_data)
    if o["question"] != s["question"]
)
print(f"Questions modified by substitution: {modified}/50")

models = {
    "Base": "saves/finetune/clinical_finetuned_v3",
    "GradDiff": "saves/unlearn/GradDiff_clinical",
    "SimNPO": "saves/unlearn/SimNPO_clinical",
    "RMU": "saves/unlearn/RMU_clinical",
    "NPO": "saves/unlearn/NPO_clinical",
}

print("\n" + "=" * 60)
print(f"{'Model':<12} {'Original':>12} {'Substituted':>12} {'Delta':>8}")
print("=" * 60)

for name, path in models.items():
    print(f"\nLoading {name}...")
    model, tokenizer = load_model(path)
    orig = evaluate_rouge(model, tokenizer, original_data, label="Original")
    sub = evaluate_rouge(model, tokenizer, substituted_data, label="Substituted")
    delta = sub - orig
    print(f"{name:<12} {orig:>12.3f} {sub:>12.3f} {delta:>+8.3f}")
    del model
    torch.cuda.empty_cache()

print("=" * 60)
print("\nInterpretation: Large delta = metrics not semantically robust")
print("Small delta = metrics are faithful to clinical synonyms")