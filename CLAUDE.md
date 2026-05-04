# OpenUnlearning Project

## What this is
Benchmarking LLM unlearning methods using the OpenUnlearning framework.
All experiments run inside a Docker container with GPU access.

## Key commands
- Train: `python src/train.py --config-name=unlearn.yaml experiment=unlearn/tofu/default forget_split=forget10 retain_split=retain90 trainer=<METHOD> task_name=<METHOD>_tofu_1B +model.load_in_4bit=true`
- Eval: `python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default forget_split=forget10 task_name=<METHOD>_tofu_1B_eval model.model_args.pretrained_model_name_or_path=saves/unlearn/<METHOD>_tofu_1B`
- Run all: `bash run_all.sh`

## Known issues
- Hydra config: use `+key=value` for new keys, `key=value` for existing keys
- Model auth: requires HuggingFace token via `huggingface-cli login`
- 4-bit quantization required for 1B model on 16GB VRAM

## Methods to benchmark
GradAscent, GradDiff, NPO, SimNPO, IDK, RMU, UNDIAL

- IDK uses `trainer=DPO` + `experiment=unlearn/tofu/idk` (not a standalone trainer)
- Available trainer configs: CEU, DPO, GradAscent, GradDiff, NPO, PDU, RMU, SatImp, SimNPO, UNDIAL, WGA, finetune

## Results location
- Checkpoints: `saves/unlearn/<METHOD>_tofu_1B/`
- Eval: `saves/eval/<METHOD>_tofu_1B_eval/TOFU_SUMMARY.json`
