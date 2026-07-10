from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from semantic_shift.config import SemanticExperimentConfig
from semantic_shift.data import TemporalBatchCollator, TemporalDiscourseDataset, build_pairs, chronological_split, load_windows
from semantic_shift.model import SemanticLearningObjective, TemporalSemanticBert


class SemanticShiftTrainer:
    def __init__(self, config: SemanticExperimentConfig):
        self.config = config
        precision = None if not torch.cuda.is_available() else config.training.mixed_precision
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision=precision,
        )
        self.output_dir = Path(config.training.output_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.model_id)
        self.model = TemporalSemanticBert(config.model, config.data.shock_names)
        self.objective = SemanticLearningObjective(
            config.training.shock_loss_weight,
            config.training.volatility_loss_weight,
            config.training.contrastive_loss_weight,
            config.training.contrastive_temperature,
        )
        self.history: list[dict] = []

    def train(self) -> list[dict]:
        self._seed()
        train_loader, validation_loader = self._loaders()
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        update_steps = math.ceil(len(train_loader) / self.config.training.gradient_accumulation_steps)
        total_steps = update_steps * self.config.training.epochs
        warmup_steps = int(total_steps * self.config.training.warmup_fraction)
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        self.model, optimizer, train_loader, validation_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, validation_loader, scheduler
        )
        best_validation = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, self.config.training.epochs + 1):
            train_metrics = self._train_epoch(train_loader, optimizer, scheduler)
            validation_metrics = self._validate(validation_loader)
            record = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}}
            self.history.append(record)
            if self.accelerator.is_main_process:
                self._write_history()
            if validation_metrics["loss"] < best_validation:
                best_validation = validation_metrics["loss"]
                epochs_without_improvement = 0
                self._save("best")
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= self.config.training.patience:
                break
        self._save("final")
        return self.history

    def _loaders(self):
        train_windows = load_windows(self.config.data.train_file, self.config.data.minimum_documents)
        train_pairs = build_pairs(train_windows)
        if self.config.data.validation_file:
            validation_windows = load_windows(self.config.data.validation_file, self.config.data.minimum_documents)
            validation_pairs = build_pairs(validation_windows)
        else:
            train_pairs, validation_pairs = chronological_split(train_pairs, self.config.data.validation_fraction)
        if not train_pairs or not validation_pairs:
            raise ValueError("Training and validation each require at least one consecutive discourse pair")
        collator = TemporalBatchCollator(self.tokenizer.pad_token_id)
        train_dataset = TemporalDiscourseDataset(train_pairs, self.tokenizer, self.config.data)
        validation_dataset = TemporalDiscourseDataset(validation_pairs, self.tokenizer, self.config.data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.training.num_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config.training.num_workers,
        )
        return train_loader, validation_loader

    def _train_epoch(self, loader, optimizer, scheduler) -> dict[str, float]:
        self.model.train()
        totals: dict[str, float] = {}
        progress = tqdm(loader, disable=not self.accelerator.is_local_main_process, desc="Semantic BERT")
        for batch in progress:
            with self.accelerator.accumulate(self.model):
                output = self.model(batch["previous"], batch["current"])
                losses = self.objective(output, batch["shock_targets"], batch["semantic_volatility"])
                self.accelerator.backward(losses["loss"])
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach())
        return {key: value / len(loader) for key, value in totals.items()}

    @torch.no_grad()
    def _validate(self, loader) -> dict[str, float]:
        self.model.eval()
        totals: dict[str, float] = {}
        for batch in loader:
            output = self.model(batch["previous"], batch["current"])
            losses = self.objective(output, batch["shock_targets"], batch["semantic_volatility"])
            for key, value in losses.items():
                gathered = self.accelerator.gather_for_metrics(value.detach().reshape(1))
                totals[key] = totals.get(key, 0.0) + float(gathered.mean())
        return {key: value / len(loader) for key, value in totals.items()}

    def _save(self, name: str) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            destination = self.output_dir / name
            model.save_pretrained(destination)
            self.tokenizer.save_pretrained(destination / "tokenizer")
            self._write_history()

    def _write_history(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in self.history) + "\n")

    def _seed(self) -> None:
        seed = self.config.training.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
