from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from semantic_shift.config import SemanticModelConfig


@dataclass
class SemanticShiftOutput:
    previous_state: torch.Tensor
    current_state: torch.Tensor
    temporal_state: torch.Tensor
    semantic_shift: torch.Tensor
    cosine_shift: torch.Tensor
    shock_embedding: torch.Tensor
    shock_values: torch.Tensor
    semantic_volatility: torch.Tensor
    predicted_current_state: torch.Tensor


class TemporalSemanticBert(nn.Module):
    def __init__(self, config: SemanticModelConfig, shock_names: tuple[str, ...]):
        super().__init__()
        self.semantic_config = config
        self.shock_names = shock_names
        self.encoder = AutoModel.from_pretrained(config.model_id)
        hidden_size = self.encoder.config.hidden_size
        if config.freeze_embeddings:
            for parameter in self.encoder.embeddings.parameters():
                parameter.requires_grad = False
        self.state_projection = nn.Sequential(
            nn.Linear(hidden_size, config.embedding_dimension),
            nn.LayerNorm(config.embedding_dimension),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        temporal_input = config.embedding_dimension * 4 + 1
        self.temporal_estimator = nn.Sequential(
            nn.Linear(temporal_input, config.embedding_dimension * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dimension * 2, config.embedding_dimension),
            nn.LayerNorm(config.embedding_dimension),
        )
        self.shock_projector = nn.Sequential(
            nn.Linear(config.embedding_dimension * 2 + 1, config.shock_embedding_dimension),
            nn.GELU(),
            nn.LayerNorm(config.shock_embedding_dimension),
        )
        self.shock_head = nn.Linear(config.shock_embedding_dimension, len(shock_names))
        self.volatility_head = nn.Sequential(
            nn.Linear(config.shock_embedding_dimension, 1),
            nn.Softplus(),
        )
        self.transition_predictor = nn.Sequential(
            nn.Linear(config.embedding_dimension, config.embedding_dimension),
            nn.GELU(),
            nn.Linear(config.embedding_dimension, config.embedding_dimension),
        )

    def encode_window(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        document_mask = batch["document_mask"]
        batch_size, documents, length = input_ids.shape
        output = self.encoder(
            input_ids=input_ids.reshape(batch_size * documents, length),
            attention_mask=attention_mask.reshape(batch_size * documents, length),
            output_hidden_states=True,
            return_dict=True,
        )
        layer_count = min(self.semantic_config.last_hidden_layers, len(output.hidden_states))
        hidden = torch.stack(output.hidden_states[-layer_count:]).mean(dim=0)
        token_mask = attention_mask.reshape(batch_size * documents, length).unsqueeze(-1)
        documents_encoded = (hidden * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp_min(1)
        documents_encoded = documents_encoded.reshape(batch_size, documents, -1)
        weights = document_mask.unsqueeze(-1).to(documents_encoded.dtype)
        window = (documents_encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.state_projection(window)

    def forward(self, previous: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> SemanticShiftOutput:
        previous_state = self.encode_window(previous)
        current_state = self.encode_window(current)
        difference = current_state - previous_state
        cosine_shift = 1.0 - F.cosine_similarity(current_state, previous_state, dim=-1)
        temporal_features = torch.cat(
            [previous_state, current_state, difference, previous_state * current_state, cosine_shift.unsqueeze(-1)],
            dim=-1,
        )
        temporal_state = self.temporal_estimator(temporal_features)
        semantic_shift = F.normalize(difference, dim=-1)
        shock_features = torch.cat([temporal_state, semantic_shift, cosine_shift.unsqueeze(-1)], dim=-1)
        shock_embedding = self.shock_projector(shock_features)
        return SemanticShiftOutput(
            previous_state=previous_state,
            current_state=current_state,
            temporal_state=temporal_state,
            semantic_shift=semantic_shift,
            cosine_shift=cosine_shift,
            shock_embedding=shock_embedding,
            shock_values=self.shock_head(shock_embedding),
            semantic_volatility=self.volatility_head(shock_embedding).squeeze(-1),
            predicted_current_state=self.transition_predictor(previous_state),
        )

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(destination / "bert_encoder")
        state = {key: value for key, value in self.state_dict().items() if not key.startswith("encoder.")}
        torch.save(state, destination / "semantic_heads.pt")
        metadata = {"model": asdict(self.semantic_config), "shock_names": list(self.shock_names)}
        (destination / "semantic_config.json").write_text(json.dumps(metadata, indent=2))

    @classmethod
    def from_pretrained(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "TemporalSemanticBert":
        source = Path(path)
        metadata = json.loads((source / "semantic_config.json").read_text())
        config = SemanticModelConfig(**metadata["model"])
        config = SemanticModelConfig(**{**asdict(config), "model_id": str(source / "bert_encoder")})
        model = cls(config, tuple(metadata["shock_names"]))
        state = torch.load(source / "semantic_heads.pt", map_location=map_location, weights_only=True)
        model.load_state_dict(state, strict=False)
        return model


class SemanticLearningObjective(nn.Module):
    def __init__(
        self,
        shock_weight: float,
        volatility_weight: float,
        contrastive_weight: float,
        temperature: float,
    ):
        super().__init__()
        self.shock_weight = shock_weight
        self.volatility_weight = volatility_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(
        self,
        output: SemanticShiftOutput,
        shock_targets: torch.Tensor,
        volatility_targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shock_mask = torch.isfinite(shock_targets)
        volatility_mask = torch.isfinite(volatility_targets)
        shock_loss = self._masked_mse(output.shock_values, shock_targets, shock_mask)
        volatility_loss = self._masked_mse(output.semantic_volatility, volatility_targets, volatility_mask)
        predicted = F.normalize(output.predicted_current_state, dim=-1)
        observed = F.normalize(output.current_state, dim=-1)
        logits = predicted @ observed.transpose(0, 1) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        total = (
            self.shock_weight * shock_loss
            + self.volatility_weight * volatility_loss
            + self.contrastive_weight * contrastive_loss
        )
        return {
            "loss": total,
            "shock_loss": shock_loss,
            "volatility_loss": volatility_loss,
            "contrastive_loss": contrastive_loss,
        }

    def _masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not mask.any():
            return prediction.sum() * 0.0
        return F.mse_loss(prediction[mask], target[mask])

