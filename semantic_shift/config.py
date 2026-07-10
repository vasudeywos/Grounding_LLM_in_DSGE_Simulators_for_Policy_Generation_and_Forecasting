from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SemanticDataConfig:
    train_file: str
    validation_file: str | None = None
    validation_fraction: float = 0.2
    max_length: int = 256
    max_documents: int = 16
    minimum_documents: int = 1
    shock_names: tuple[str, ...] = (
        "technology",
        "demand",
        "supply",
        "expectations",
        "inflation",
        "monetary",
    )


@dataclass(frozen=True)
class SemanticModelConfig:
    model_id: str = "bert-base-uncased"
    embedding_dimension: int = 256
    shock_embedding_dimension: int = 128
    last_hidden_layers: int = 4
    dropout: float = 0.1
    freeze_embeddings: bool = False


@dataclass(frozen=True)
class SemanticTrainingConfig:
    output_dir: str = "outputs/semantic_bert"
    seed: int = 42
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.00002
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    patience: int = 3
    shock_loss_weight: float = 1.0
    volatility_loss_weight: float = 0.5
    contrastive_loss_weight: float = 0.2
    contrastive_temperature: float = 0.07
    mixed_precision: str = "bf16"
    num_workers: int = 0


@dataclass(frozen=True)
class SemanticExperimentConfig:
    data: SemanticDataConfig
    model: SemanticModelConfig = field(default_factory=SemanticModelConfig)
    training: SemanticTrainingConfig = field(default_factory=SemanticTrainingConfig)


def _resolve(base: Path, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str((path if path.is_absolute() else base / path).resolve())


def load_semantic_config(path: str | Path) -> SemanticExperimentConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text())
    data_raw = raw["data"]
    data_raw["train_file"] = _resolve(source.parent, data_raw["train_file"])
    data_raw["validation_file"] = _resolve(source.parent, data_raw.get("validation_file"))
    if "shock_names" in data_raw:
        data_raw["shock_names"] = tuple(data_raw["shock_names"])
    data = SemanticDataConfig(**data_raw)
    model = SemanticModelConfig(**raw.get("model", {}))
    training = SemanticTrainingConfig(**raw.get("training", {}))
    return SemanticExperimentConfig(data=data, model=model, training=training)

