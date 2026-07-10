from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass(frozen=True)
class TargetConfig:
    variable: str
    label: str
    target: float = 0.0
    weight: float = 1.0


@dataclass(frozen=True)
class LeverConfig:
    name: str
    shock: str
    minimum: float
    maximum: float
    persistence: float = 0.8


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    shocks: dict[str, list[float]]
    semantic_volatility: float = 0.0
    discourse: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentConfig:
    model_path: str
    periods: int
    burn_in: int
    turns: int
    action_duration: int
    targets: tuple[TargetConfig, ...]
    levers: tuple[LeverConfig, ...]
    scenarios: tuple[ScenarioConfig, ...]
    volatility_weight: float = 0.0
    action_weight: float = 0.01
    invalid_action_penalty: float = 2.0
    reward_scale: float = 10.0


@dataclass(frozen=True)
class ModelConfig:
    model_id: str = MODEL_ID
    load_in_4bit: bool = True
    torch_dtype: str = "bfloat16"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    seed: int = 42
    epochs: int = 3
    learning_rate: float = 5e-6
    max_new_tokens: int = 80
    temperature: float = 1.0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_every: int = 25
    group_size: int = 4
    grpo_epochs: int = 1
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    environment: EnvironmentConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=lambda: TrainingConfig(output_dir="outputs"))


def _tuple_of(items: list[dict[str, Any]], cls: type) -> tuple:
    return tuple(cls(**item) for item in items)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text())
    env = raw["environment"]
    model_path = Path(env["model_path"])
    if not model_path.is_absolute():
        model_path = source.parent / model_path
    environment = EnvironmentConfig(
        model_path=str(model_path.resolve()),
        periods=int(env["periods"]),
        burn_in=int(env["burn_in"]),
        turns=int(env["turns"]),
        action_duration=int(env["action_duration"]),
        targets=_tuple_of(env["targets"], TargetConfig),
        levers=_tuple_of(env["levers"], LeverConfig),
        scenarios=_tuple_of(env["scenarios"], ScenarioConfig),
        volatility_weight=float(env.get("volatility_weight", 0.0)),
        action_weight=float(env.get("action_weight", 0.01)),
        invalid_action_penalty=float(env.get("invalid_action_penalty", 2.0)),
        reward_scale=float(env.get("reward_scale", 10.0)),
    )
    model_raw = raw.get("model", {})
    if "lora_targets" in model_raw:
        model_raw["lora_targets"] = tuple(model_raw["lora_targets"])
    model = ModelConfig(**model_raw)
    training = TrainingConfig(**raw.get("training", {"output_dir": "outputs"}))
    return ExperimentConfig(environment=environment, model=model, training=training)
