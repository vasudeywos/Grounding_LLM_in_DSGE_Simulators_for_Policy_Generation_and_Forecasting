from __future__ import annotations

from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dsge_rl.config import ModelConfig


def resolve_dtype(name: str) -> torch.dtype:
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unknown torch dtype {name}")
    return value


def _load_lora_model(config: ModelConfig):
    dtype = resolve_dtype(config.torch_dtype)
    quantization = None
    if config.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
    )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.lora_targets),
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def load_policy(config: ModelConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    return _load_lora_model(config), tokenizer


class TokenValueModel(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False, device=backbone.device, dtype=backbone.dtype)

    @property
    def device(self):
        return self.backbone.device

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=False,
        )
        values = self.value_head(output.hidden_states[-1]).squeeze(-1)
        return values

    def save_pretrained(self, path: str):
        self.backbone.save_pretrained(path)
        torch.save(self.value_head.state_dict(), f"{path}/value_head.pt")


def load_value_model(config: ModelConfig) -> TokenValueModel:
    return TokenValueModel(_load_lora_model(config))


def load_adapter_policy(config: ModelConfig, adapter_path: str | Path):
    dtype = resolve_dtype(config.torch_dtype)
    quantization = None
    if config.load_in_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    return AutoPeftModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=dtype,
        quantization_config=quantization,
        device_map="auto",
        is_trainable=False,
    )


def token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    shifted = torch.log_softmax(logits[:, :-1], dim=-1)
    return shifted.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
