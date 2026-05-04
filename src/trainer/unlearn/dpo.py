import torch
import torch.nn.functional as F
from torch import nn
from trainer.unlearn.grad_diff import GradDiff
from trainer.utils import compute_batch_nll


def _compute_batch_nll_cpu(ref_model, inputs):
    """Run ref_model on CPU, return losses on original device."""
    device = next(iter(inputs.values())).device
    cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ref_model(**cpu_inputs)
    logits = outputs.logits
    labels = cpu_inputs["labels"]
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    loss = loss_fn(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss.to(device)


def _compute_dpo_loss_cpu_ref(model, ref_model, win_inputs=None, lose_inputs=None, beta=1.0):
    """DPO loss using CPU-offloaded ref_model."""
    win_log_ratio, lose_log_ratio = 0.0, 0.0
    win_outputs, lose_outputs = None, None

    if win_inputs is not None:
        win_loss, win_outputs = compute_batch_nll(model, win_inputs)
        win_ref_loss = _compute_batch_nll_cpu(ref_model, win_inputs)
        win_log_ratio = -(win_loss - win_ref_loss)

    if lose_inputs is not None:
        lose_loss, lose_outputs = compute_batch_nll(model, lose_inputs)
        lose_ref_loss = _compute_batch_nll_cpu(ref_model, lose_inputs)
        lose_log_ratio = -(lose_loss - lose_ref_loss)

    loss = -2 / beta * F.logsigmoid(beta * (win_log_ratio - lose_log_ratio)).mean()
    return loss, (win_outputs, lose_outputs)


class DPO(GradDiff):
    def __init__(self, beta=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model_cpu(self.model)

    def _prepare_ref_model_cpu(self, model):
        """Load ref model in bf16 on CPU with eager attention (no CUDA required)."""
        from transformers import AutoModelForCausalLM
        model_name = model.config._name_or_path
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
        ref_model.eval()
        return ref_model

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        forget_inputs = inputs["forget"]["original"]
        alternate_inputs = inputs["forget"]["alternate"]

        forget_loss, forget_outputs = _compute_dpo_loss_cpu_ref(
            model=model,
            ref_model=self.ref_model,
            win_inputs=alternate_inputs,
            lose_inputs=forget_inputs,
            beta=self.beta,
        )

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)

        loss = self.gamma * forget_loss + self.alpha * retain_loss
        return (loss, forget_outputs) if return_outputs else loss
