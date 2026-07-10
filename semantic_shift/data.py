from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from semantic_shift.config import SemanticDataConfig


@dataclass(frozen=True)
class DiscourseWindow:
    timestamp: str
    source: str
    texts: tuple[str, ...]
    shock_targets: dict[str, float]
    semantic_volatility: float | None


def load_windows(path: str | Path, minimum_documents: int = 1) -> list[DiscourseWindow]:
    windows = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        texts = row.get("texts")
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list) or len(texts) < minimum_documents:
            continue
        timestamp = str(row["timestamp"])
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        windows.append(
            DiscourseWindow(
                timestamp=timestamp,
                source=str(row.get("source", "unknown")),
                texts=tuple(str(text) for text in texts if str(text).strip()),
                shock_targets={str(key): float(value) for key, value in row.get("shock_targets", {}).items()},
                semantic_volatility=float(row["semantic_volatility"]) if row.get("semantic_volatility") is not None else None,
            )
        )
    windows.sort(key=lambda item: (item.source, item.timestamp))
    return windows


def build_pairs(windows: list[DiscourseWindow]) -> list[tuple[DiscourseWindow, DiscourseWindow]]:
    pairs = []
    by_source: dict[str, list[DiscourseWindow]] = {}
    for window in windows:
        by_source.setdefault(window.source, []).append(window)
    for source_windows in by_source.values():
        pairs.extend(zip(source_windows[:-1], source_windows[1:]))
    return sorted(pairs, key=lambda pair: pair[1].timestamp)


def chronological_split(pairs, validation_fraction: float):
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    split = max(1, min(len(pairs) - 1, int(len(pairs) * (1.0 - validation_fraction))))
    return pairs[:split], pairs[split:]


class TemporalDiscourseDataset(Dataset):
    def __init__(self, pairs, tokenizer, config: SemanticDataConfig):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer
        self.config = config

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        previous, current = self.pairs[index]
        return {
            "previous": self._encode(previous.texts),
            "current": self._encode(current.texts),
            "shock_targets": torch.tensor(
                [current.shock_targets.get(name, float("nan")) for name in self.config.shock_names],
                dtype=torch.float32,
            ),
            "semantic_volatility": torch.tensor(
                current.semantic_volatility if current.semantic_volatility is not None else float("nan"),
                dtype=torch.float32,
            ),
            "timestamp": current.timestamp,
            "source": current.source,
        }

    def _encode(self, texts: tuple[str, ...]) -> dict[str, torch.Tensor]:
        selected = list(texts[: self.config.max_documents])
        encoded = self.tokenizer(
            selected,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {key: value for key, value in encoded.items()}


class TemporalBatchCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict]) -> dict:
        return {
            "previous": self._pad_documents([item["previous"] for item in items]),
            "current": self._pad_documents([item["current"] for item in items]),
            "shock_targets": torch.stack([item["shock_targets"] for item in items]),
            "semantic_volatility": torch.stack([item["semantic_volatility"] for item in items]),
            "timestamp": [item["timestamp"] for item in items],
            "source": [item["source"] for item in items],
        }

    def _pad_documents(self, groups: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        maximum = max(group["input_ids"].shape[0] for group in groups)
        result: dict[str, list[torch.Tensor]] = {}
        document_masks = []
        for group in groups:
            count, length = group["input_ids"].shape
            document_masks.append(torch.cat([torch.ones(count), torch.zeros(maximum - count)]))
            for key, value in group.items():
                pad_value = self.pad_token_id if key == "input_ids" else 0
                padding = torch.full((maximum - count, length), pad_value, dtype=value.dtype)
                result.setdefault(key, []).append(torch.cat([value, padding], dim=0))
        batch = {key: torch.stack(values) for key, values in result.items()}
        batch["document_mask"] = torch.stack(document_masks)
        return batch

