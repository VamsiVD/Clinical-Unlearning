# Benchmarking Medical Knowledge Erasure
### Applying the OpenUnlearning Framework to Clinical LLMs

**Team:** Justin Chen, Vamsi Dandu, Aayet Hashmi
**Course:** NLP Spring 2026

---

## Overview

This project benchmarks machine unlearning methods on clinical patient data. We fine-tune LLaMA-3.2-1B on synthetic HIPAA-style patient vignettes to create a model that genuinely memorizes patient data, then apply four unlearning methods (GradDiff, SimNPO, RMU, NPO) and evaluate forgetting and utility preservation.

---

## Repository Structure

```
├── clinical_eval.py              # Main evaluation script
├── clinical_eval_final.py        # Full eval including relearning attacks
├── clinical_forget_set_final.json # 326 synthetic patient vignettes (forget set)
├── retain_set.json               # 400 MedQA entries (retain set)
├── configs/
│   ├── data/datasets/
│   │   ├── Clinical_forget.yaml  # HuggingFace dataset config for forget set
│   │   └── Clinical_retain.yaml  # HuggingFace dataset config for retain set
│   └── experiment/
│       ├── finetune/clinical/default.yaml  # Fine-tuning config
│       └── unlearn/clinical/default.yaml   # Unlearning config
└── README.md
```

---

## Setup

### Prerequisites
- Docker (recommended) or Python 3.11+
- NVIDIA GPU with 16GB+ VRAM
- HuggingFace account

### Installation

```bash
# Clone OpenUnlearning
git clone https://github.com/locuslab/open-unlearning.git
cd open-unlearning
pip install ".[lm-eval]"
pip install rouge-score

# HuggingFace login (required for LLaMA access)
unset HF_TOKEN
huggingface-cli login
```

### Fix BFloat16 bug (required)
```bash
sed -i 's/avg_losses = avg_losses.cpu().numpy().tolist()/avg_losses = avg_losses.cpu().float().numpy().tolist()/g' \
    src/evals/metrics/utils.py
sed -i 's/normalized_probs = normalized_probs.cpu().numpy().tolist()/normalized_probs = normalized_probs.cpu().float().numpy().tolist()/g' \
    src/evals/metrics/utils.py
```

---

## Quick Demo (Verify Pipeline — ~5 minutes)

Run this to verify the evaluation pipeline works on 10 samples:

```python
# demo.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from datasets import load_dataset
import torch

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="cuda",
    load_in_4bit=True
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

print("Loading dataset...")
ds = load_dataset("Cosmic148/clinical-unlearning", data_files="forget.json", split="train")
data = list(ds)[:10]

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
scores = []

print("Evaluating 10 samples...")
for item in data:
    inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    score = scorer.score(item["answer"], generated)["rougeL"].fmeasure
    scores.append(score)
    print(f"  Q: {item['question'][:60]}...")
    print(f"  Generated: {generated[:80]}...")
    print(f"  ROUGE-L: {score:.3f}")
    print()

print(f"Average ROUGE-L (base model, no unlearning): {sum(scores)/len(scores):.3f}")
print("Expected: ~0.15-0.20 for base LLaMA without clinical fine-tuning")
```

```bash
python demo.py
```

---

## Full Experiment Reproduction

### Step 1 — Fine-tune LLaMA on clinical data (creates memorized model)

```bash
python src/train.py --config-name=train.yaml \
  experiment=finetune/clinical/default \
  task_name=clinical_finetuned \
  trainer.args.num_train_epochs=15 \
  trainer.args.learning_rate=5e-5 \
  +model.load_in_4bit=true
```

Output: `saves/finetune/clinical_finetuned/`
Expected forget ROUGE-L: ~0.55-0.65 (confirms memorization)

### Step 2 — Run unlearning methods

```bash
for METHOD in GradDiff SimNPO RMU NPO; do
  python src/train.py --config-name=unlearn.yaml \
    experiment=unlearn/clinical/default \
    trainer=${METHOD} \
    task_name=${METHOD}_clinical \
    +model.load_in_4bit=true
done
```

Output: `saves/unlearn/<METHOD>_clinical/`

### Step 3 — Run relearning attacks

```bash
for METHOD in GradDiff SimNPO RMU NPO; do
  python src/train.py --config-name=train.yaml \
    experiment=finetune/clinical/default \
    trainer.args.num_train_epochs=1 \
    trainer.args.learning_rate=5e-5 \
    task_name=${METHOD}_clinical_relearn \
    model.model_args.pretrained_model_name_or_path=./saves/unlearn/${METHOD}_clinical \
    +model.load_in_4bit=true
done
```

### Step 4 — Evaluate all models

```bash
python clinical_eval_final.py
```

Results saved to `clinical_final_results.json`

---

## Dataset

**Forget set:** `Cosmic148/clinical-unlearning` → `forget.json`
- 326 unique synthetic HIPAA-style patient vignettes
- Completely fictional patient names and clinical details
- 6 disease categories: cardiac, oncology, neurology, infectious, endocrine, pulmonary
- Format: `{"question": "What was the diagnosis and treatment for patient [Name]...", "answer": "..."}`

**Retain set:** `Cosmic148/clinical-unlearning` → `retain.json`
- 400 MedQA USMLE-style general clinical QA pairs
- No patient-specific information
- Tests preservation of general medical knowledge after unlearning

---

## Results

| Model | Forget ROUGE-L↓ | Retain ROUGE-L↑ | Forgetting |
|---|---|---|---|
| Base clinical | 0.552 | 0.016 | — |
| GradDiff | 0.175 | 0.030 | -68.3% |
| SimNPO | 0.179 | 0.018 | -67.6% |
| RMU | 0.171 | 0.019 | -69.0% |
| NPO | 0.203 | 0.021 | -63.2% |

### Relearning Attack

| Model | Before | After relearn | Recovery |
|---|---|---|---|
| GradDiff | 0.175 | 0.244 | +39.4% |
| SimNPO | 0.179 | 0.211 | +17.9% |
| RMU | 0.171 | 0.251 | +46.8% |
| NPO | 0.203 | 0.233 | +14.8% |

---

## Compute Resources

| Hardware | Usage |
|---|---|
| NVIDIA RTX 4070 Ti Super (16GB) | Fine-tuning + unlearning + evaluation |
| Google Colab A100 (40GB) | Additional runs |

---

## Citation

```bibtex
@article{dorna2025openunlearning,
  title={OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking},
  author={Dorna, Vineeth and others},
  journal={arXiv preprint arXiv:2506.12618},
  year={2025}
}
```
