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
from dsge_rl.trainers.common import disable_dropout, save_metrics, set_seed, trainable_parameters


class GRPOTrainer:
    def __init__(self, model, tokenizer, environment: DSGEPolicyEnvironment, config: ExperimentConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.environment = environment
        self.config = config
        self.optimizer = torch.optim.AdamW(trainable_parameters(model), lr=config.training.learning_rate)
        self.output_dir = Path(config.training.output_dir)
        self.metrics: list[dict] = []
        disable_dropout(self.model)

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
                old_log_probs = self._behavior_log_probabilities(trajectories)
                reference_log_probs = self._reference_log_probabilities(trajectories)
                loss, mean_kl, clip_fraction = self._update(
                    trajectories,
                    advantages,
                    old_log_probs,
                    reference_log_probs,
                )
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
                    "reference_kl": mean_kl,
                    "clip_fraction": clip_fraction,
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

    def _behavior_log_probabilities(self, trajectories: list[list[TurnRollout]]) -> list[torch.Tensor]:
        results = []
        for trajectory in trajectories:
            values = []
            for turn in trajectory:
                if turn.behavior_log_probs is None:
                    raise RuntimeError("Rollout is missing generation-time behavior log probabilities")
                values.append(turn.behavior_log_probs.to(self.model.device))
            results.append(torch.cat(values).detach())
        return results

    def _log_probabilities(self, trajectories: list[list[TurnRollout]]) -> list[torch.Tensor]:
        results = []
        for trajectory in trajectories:
            selected = []
            for turn in trajectory:
                logits = self.model(input_ids=turn.input_ids.unsqueeze(0), use_cache=False).logits
                log_probs = token_log_probs(logits, turn.input_ids.unsqueeze(0))[0]
                selected.append(log_probs[turn.action_mask])
            results.append(torch.cat(selected))
        return results

    def _reference_log_probabilities(self, trajectories: list[list[TurnRollout]]) -> list[torch.Tensor] | None:
        if self.config.training.grpo_kl_coefficient == 0.0:
            return None
        disable_adapter = getattr(self.model, "disable_adapter", None)
        if disable_adapter is None:
            raise RuntimeError("GRPO KL requires a PEFT policy with disable_adapter support")
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad(), disable_adapter():
            values = self._log_probabilities(trajectories)
            values = [value.detach() for value in values]
        if was_training:
            self.model.train()
        return values

    def _update(
        self,
        trajectories: list[list[TurnRollout]],
        advantages: torch.Tensor,
        old_log_probs: list[torch.Tensor],
        reference_log_probs: list[torch.Tensor] | None,
    ) -> tuple[float, float, float]:
        self.model.train()
        final_loss = 0.0
        final_kl = 0.0
        final_clip_fraction = 0.0
        for _ in range(self.config.training.grpo_epochs):
            current_log_probs = self._log_probabilities(trajectories)
            losses = []
            kls = []
            clipped_tokens = []
            for index, (current, old, advantage) in enumerate(zip(current_log_probs, old_log_probs, advantages)):
                ratio = torch.exp(current - old)
                unclipped = ratio * advantage
                clipped_ratio = ratio.clamp(
                    1 - self.config.training.clip_range,
                    1 + self.config.training.clip_range,
                )
                clipped = clipped_ratio * advantage
                policy_loss = -torch.minimum(unclipped, clipped)
                if reference_log_probs is None:
                    kl = torch.zeros_like(current)
                else:
                    log_ratio = reference_log_probs[index] - current
                    kl = torch.exp(log_ratio) - log_ratio - 1.0
                losses.append((policy_loss + self.config.training.grpo_kl_coefficient * kl).mean())
                kls.append(kl.mean())
                clipped_tokens.append((ratio != clipped_ratio).float().mean())
            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(trainable_parameters(self.model), self.config.training.max_grad_norm)
            self.optimizer.step()
            final_loss = float(loss.detach())
            final_kl = float(torch.stack(kls).mean().detach())
            final_clip_fraction = float(torch.stack(clipped_tokens).mean().detach())
        return final_loss, final_kl, final_clip_fraction

    def _save(self, step: int, final: bool = False) -> None:
        destination = self.output_dir / ("final" if final else f"checkpoint-{step}")
        destination.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(destination))
        self.tokenizer.save_pretrained(str(destination))
        save_metrics(self.output_dir / "metrics.jsonl", self.metrics)
