from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torch import nn
from torch.nn import functional as F
from transformers import BitsAndBytesConfig

from dsge_rl.modeling import TokenValueModel
from dsge_rl.modeling import load_adapter_policy
from dsge_rl.config import ModelConfig


SFR_MODEL_ID = "Salesforce/SFR-Embedding-2_R"


@dataclass(frozen=True)
class SFRConfig:
    model_id: str = SFR_MODEL_ID
    max_sequence_length: int = 4096
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"
    bridge_hidden_dimension: int = 1024
    bridge_dropout: float = 0.1


class SFRShiftEncoder:
    def __init__(self, config: SFRConfig):
        self.config = config
        dtype = getattr(torch, config.torch_dtype)
        model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        if config.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        self.model = SentenceTransformer(config.model_id, model_kwargs=model_kwargs)
        self.model.max_seq_length = config.max_sequence_length
        self.dimension = int(self.model.get_sentence_embedding_dimension()) * 2 + 1
        self.cache: dict[tuple[str, str], torch.Tensor] = {}

    @torch.no_grad()
    def encode_shift(self, previous: str, current: str) -> torch.Tensor:
        key = (previous, current)
        if key not in self.cache:
            states = self.model.encode(
                [previous, current],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).float()
            difference = states[1] - states[0]
            cosine_shift = 1.0 - F.cosine_similarity(states[1].unsqueeze(0), states[0].unsqueeze(0)).reshape(1)
            self.cache[key] = torch.cat([states[1], difference, cosine_shift]).cpu()
        return self.cache[key].clone()


class SemanticBridge(nn.Module):
    def __init__(self, input_dimension: int, output_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.input_dimension = input_dimension
        self.network = nn.Sequential(
            nn.LayerNorm(input_dimension),
            nn.Linear(input_dimension, hidden_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, output_dimension),
            nn.LayerNorm(output_dimension),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate) * self.network(semantic_features)


class SFRConditionedPolicy(nn.Module):
    def __init__(self, policy: nn.Module, sfr_dimension: int, config: SFRConfig):
        super().__init__()
        self.policy = policy
        self.sfr_config = config
        self.bridge = SemanticBridge(
            sfr_dimension,
            policy.config.hidden_size,
            config.bridge_hidden_dimension,
            config.bridge_dropout,
        ).to(device=policy.device, dtype=policy.dtype)

    @property
    def device(self):
        return self.policy.device

    def _embeddings(self, input_ids: torch.Tensor, semantic_features: torch.Tensor) -> torch.Tensor:
        token_embeddings = self.policy.get_input_embeddings()(input_ids)
        features = semantic_features.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        conditioning = self.bridge(features).unsqueeze(1)
        return token_embeddings + conditioning

    def forward(self, input_ids: torch.Tensor, semantic_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        embeddings = self._embeddings(input_ids, semantic_features)
        return self.policy(inputs_embeds=embeddings, attention_mask=attention_mask, use_cache=False)

    def generate(self, input_ids: torch.Tensor, semantic_features: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        embeddings = self._embeddings(input_ids, semantic_features)
        return self.policy.generate(
            input_ids=input_ids,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            **kwargs,
        )

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(destination / "policy_adapter")
        torch.save(self.bridge.state_dict(), destination / "semantic_bridge.pt")
        (destination / "sfr_config.json").write_text(json.dumps(asdict(self.sfr_config), indent=2))

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        sfr_dimension: int,
        model_config: ModelConfig,
    ) -> "SFRConditionedPolicy":
        source = Path(path)
        sfr_config = SFRConfig(**json.loads((source / "sfr_config.json").read_text()))
        policy = load_adapter_policy(model_config, source / "policy_adapter")
        model = cls(policy, sfr_dimension, sfr_config)
        state = torch.load(source / "semantic_bridge.pt", map_location=model.device, weights_only=True)
        model.bridge.load_state_dict(state)
        return model


class SFRConditionedValueModel(nn.Module):
    def __init__(self, value_model: TokenValueModel, sfr_dimension: int, config: SFRConfig):
        super().__init__()
        self.value_model = value_model
        self.bridge = SemanticBridge(
            sfr_dimension,
            value_model.backbone.config.hidden_size,
            config.bridge_hidden_dimension,
            config.bridge_dropout,
        ).to(device=value_model.device, dtype=value_model.backbone.dtype)

    @property
    def device(self):
        return self.value_model.device

    def forward(self, input_ids: torch.Tensor, semantic_features: torch.Tensor, attention_mask: torch.Tensor | None = None):
        token_embeddings = self.value_model.backbone.get_input_embeddings()(input_ids)
        features = semantic_features.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
        embeddings = token_embeddings + self.bridge(features).unsqueeze(1)
        return self.value_model(inputs_embeds=embeddings, attention_mask=attention_mask)

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.value_model.save_pretrained(str(destination / "value_adapter"))
        torch.save(self.bridge.state_dict(), destination / "semantic_bridge.pt")
