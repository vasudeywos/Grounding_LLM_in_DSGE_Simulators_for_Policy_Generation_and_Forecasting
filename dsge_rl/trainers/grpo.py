from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dsge_rl.config import ExperimentConfig
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import token_log_probs
from dsge_rl.rollout import TurnRollout, collect_trajectory
from dsge_rl.trainers.common import save_metrics, set_seed, trainable_parameters


class GRPOTrainer:
    def __init__(self, model, tokenizer, environment: DSGEPolicyEnvironment, config: ExperimentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.environment = environment
        self.config = config
        self.optimizer = torch.optim.AdamW(trainable_parameters(model), lr=config.training.learning_rate)
        self.output_dir = Path(config.training.output_dir)
        self.metrics: list[dict] = []

    def train(self) -> list[dict]:
        set_seed(self.config.training.seed)
        scenarios = list(self.config.environment.scenarios)
        total = self.config.training.epochs * len(scenarios)
        progress = tqdm(total=total, desc="GRPO")
        step = 0
        for epoch in range(self.config.training.epochs):
            np.random.shuffle(scenarios)
            for scenario in scenarios:
                trajectories = [self._rollout(scenario) for _ in range(self.config.training.group_size)]
                rewards = torch.tensor(
                    [sum(turn.reward for turn in trajectory) for trajectory in trajectories],
                    dtype=torch.float32,
                    device=self.model.device,
                )
                standard_deviation = rewards.std(unbiased=False)
                advantages = (rewards - rewards.mean()) / (standard_deviation + 1e-8)
                old_log_probs = self._log_probabilities(trajectories, detached=True)
                loss = self._update(trajectories, advantages, old_log_probs)
                zero_signal = bool(standard_deviation.item() < 1e-8)
                record = {
                    "algorithm": "grpo",
                    "epoch": epoch + 1,
                    "step": step + 1,
                    "scenario": scenario.name,
                    "reward_mean": rewards.mean().item(),
                    "reward_std": standard_deviation.item(),
                    "zero_advantage_group": zero_signal,
                    "loss": loss,
                }
                self.metrics.append(record)
                step += 1
                progress.update(1)
                progress.set_postfix(reward=f"{record['reward_mean']:.3f}", zero=zero_signal)
                if step % self.config.training.save_every == 0:
                    self._save(step)
        progress.close()
        self._save(step, final=True)
        return self.metrics

    def _rollout(self, scenario) -> list[TurnRollout]:
        return collect_trajectory(
            self.model,
            self.tokenizer,
            self.environment.clone(),
            scenario,
            self.config.training.max_new_tokens,
            self.config.training.temperature,
        )

    def _log_probabilities(self, trajectories: list[list[TurnRollout]], detached: bool) -> list[torch.Tensor]:
        results = []
        context = torch.no_grad() if detached else torch.enable_grad()
        with context:
            for trajectory in trajectories:
                selected = []
                for turn in trajectory:
                    logits = self.model(input_ids=turn.input_ids.unsqueeze(0), use_cache=False).logits
                    log_probs = token_log_probs(logits, turn.input_ids.unsqueeze(0))[0]
                    selected.append(log_probs[turn.action_mask])
                results.append(torch.cat(selected))
        return results

    def _update(self, trajectories: list[list[TurnRollout]], advantages: torch.Tensor, old_log_probs: list[torch.Tensor]) -> float:
        self.model.train()
        final_loss = 0.0
        for _ in range(self.config.training.grpo_epochs):
            current_log_probs = self._log_probabilities(trajectories, detached=False)
            losses = []
            for current, old, advantage in zip(current_log_probs, old_log_probs, advantages):
                ratio = torch.exp(current - old)
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1 - self.config.training.clip_range,
                    1 + self.config.training.clip_range,
                ) * advantage
                losses.append(-torch.minimum(unclipped, clipped).mean())
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trainable_parameters(self.model), self.config.training.max_grad_norm)
            self.optimizer.step()
            final_loss = float(loss.detach())
        return final_loss

    def _save(self, step: int, final: bool = False) -> None:
        destination = self.output_dir / ("final" if final else f"checkpoint-{step}")
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(str(destination))
        save_metrics(self.output_dir / "metrics.jsonl", self.metrics)
