from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dsge_rl.config import ExperimentConfig
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import TokenValueModel, token_log_probs
from dsge_rl.rollout import collect_trajectory
from dsge_rl.trainers.common import save_metrics, set_seed, trainable_parameters


class PPOTrainer:
    def __init__(self, model, value_model: TokenValueModel, tokenizer, environment: DSGEPolicyEnvironment, config: ExperimentConfig):
        self.model = model
        self.value_model = value_model
        self.tokenizer = tokenizer
        self.environment = environment
        self.config = config
        self.policy_optimizer = torch.optim.AdamW(trainable_parameters(model), lr=config.training.learning_rate)
        self.value_optimizer = torch.optim.AdamW(trainable_parameters(value_model), lr=config.training.learning_rate)
        self.output_dir = Path(config.training.output_dir)
        self.metrics: list[dict] = []

    def train(self) -> list[dict]:
        set_seed(self.config.training.seed)
        scenarios = list(self.config.environment.scenarios)
        total = self.config.training.epochs * len(scenarios)
        progress = tqdm(total=total, desc="PPO")
        step = 0
        for epoch in range(self.config.training.epochs):
            np.random.shuffle(scenarios)
            for scenario in scenarios:
                trajectory = collect_trajectory(
                    self.model,
                    self.tokenizer,
                    self.environment.clone(),
                    scenario,
                    self.config.training.max_new_tokens,
                    self.config.training.temperature,
                )
                batch = self._prepare(trajectory)
                policy_loss, value_loss = self._update(batch)
                trajectory_return = sum(turn.reward for turn in trajectory)
                record = {
                    "algorithm": "ppo",
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "scenario": scenario.name,
                    "trajectory_return": trajectory_return,
                    "turn_rewards": [turn.reward for turn in trajectory],
                    "advantage_nonzero_fraction": float((batch["advantages"].abs() > 1e-8).float().mean()),
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                }
                self.metrics.append(record)
                step += 1
                progress.update(1)
                progress.set_postfix(reward=f"{trajectory_return:.3f}")
                if step % self.config.training.save_every == 0:
                    self._save(step)
        progress.close()
        self._save(step, final=True)
        return self.metrics

    def _prepare(self, trajectory) -> dict[str, torch.Tensor | list]:
        old_log_probs = []
        old_values = []
        segments = []
        with torch.no_grad():
            for turn in trajectory:
                ids = turn.input_ids.unsqueeze(0)
                logits = self.model(input_ids=ids, use_cache=False).logits
                values = self.value_model(ids)
                chosen_log_probs = token_log_probs(logits, ids)[0][turn.action_mask]
                chosen_values = values[:, :-1][0][turn.action_mask]
                old_log_probs.append(chosen_log_probs)
                old_values.append(chosen_values)
                segments.append((turn.input_ids, turn.action_mask))
        log_probs = torch.cat(old_log_probs)
        values = torch.cat(old_values)
        rewards = torch.zeros_like(values)
        offset = 0
        for turn, segment in zip(trajectory, old_values):
            offset += len(segment)
            rewards[offset - 1] = turn.reward
        advantages = self._gae(rewards, values)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        return {
            "segments": segments,
            "old_log_probs": log_probs,
            "old_values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        advantages = torch.zeros_like(rewards)
        running = torch.zeros((), device=rewards.device)
        next_value = torch.zeros((), device=rewards.device)
        for index in range(len(rewards) - 1, -1, -1):
            delta = rewards[index] + self.config.training.gamma * next_value - values[index]
            running = delta + self.config.training.gamma * self.config.training.gae_lambda * running
            advantages[index] = running
            next_value = values[index]
        return advantages

    def _current(self, segments) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = []
        values = []
        for input_ids, action_mask in segments:
            ids = input_ids.unsqueeze(0)
            logits = self.model(input_ids=ids, use_cache=False).logits
            predicted_values = self.value_model(ids)
            probabilities.append(token_log_probs(logits, ids)[0][action_mask])
            values.append(predicted_values[:, :-1][0][action_mask])
        return torch.cat(probabilities), torch.cat(values)

    def _update(self, batch: dict) -> tuple[float, float]:
        final_policy_loss = 0.0
        final_value_loss = 0.0
        for _ in range(self.config.training.ppo_epochs):
            log_probs, values = self._current(batch["segments"])
            ratio = torch.exp(log_probs - batch["old_log_probs"])
            unclipped = ratio * batch["advantages"]
            clipped = ratio.clamp(1 - self.config.training.clip_range, 1 + self.config.training.clip_range) * batch["advantages"]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            clipped_values = batch["old_values"] + (values - batch["old_values"]).clamp(
                -self.config.training.value_clip_range,
                self.config.training.value_clip_range,
            )
            value_loss = 0.5 * torch.maximum(
                F.mse_loss(values, batch["returns"], reduction="none"),
                F.mse_loss(clipped_values, batch["returns"], reduction="none"),
            ).mean()
            entropy_proxy = -log_probs.mean()
            policy_objective = policy_loss - self.config.training.entropy_coefficient * entropy_proxy
            value_objective = self.config.training.value_coefficient * value_loss
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            policy_objective.backward()
            value_objective.backward()
            clip_grad_norm_(trainable_parameters(self.model), self.config.training.max_grad_norm)
            clip_grad_norm_(trainable_parameters(self.value_model), self.config.training.max_grad_norm)
            self.policy_optimizer.step()
            self.value_optimizer.step()
            final_policy_loss = float(policy_loss.detach())
            final_value_loss = float(value_loss.detach())
        return final_policy_loss, final_value_loss

    def _save(self, step: int, final: bool = False) -> None:
        destination = self.output_dir / ("final" if final else f"checkpoint-{step}")
        destination.mkdir(parents=True, exist_ok=True)
        policy_destination = destination / "policy"
        value_destination = destination / "value_model"
        policy_destination.mkdir(parents=True, exist_ok=True)
        value_destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(policy_destination))
        self.value_model.save_pretrained(str(value_destination))
        self.tokenizer.save_pretrained(str(policy_destination))
        save_metrics(self.output_dir / "metrics.jsonl", self.metrics)
