import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from semantic_shift.config import SemanticDataConfig
from semantic_shift.data import TemporalBatchCollator, TemporalDiscourseDataset, build_pairs, load_windows
from semantic_shift.model import TemporalSemanticBert


def _move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-documents", type=int, default=16)
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalSemanticBert.from_pretrained(checkpoint, map_location=device).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "tokenizer")
    config = SemanticDataConfig(
        train_file=args.input,
        max_length=args.max_length,
        max_documents=args.max_documents,
        shock_names=model.shock_names,
    )
    pairs = build_pairs(load_windows(args.input))
    dataset = TemporalDiscourseDataset(pairs, tokenizer, config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=TemporalBatchCollator(tokenizer.pad_token_id),
    )
    rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(_move(batch["previous"], device), _move(batch["current"], device))
            for index, timestamp in enumerate(batch["timestamp"]):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "source": batch["source"][index],
                        "shock_embedding": output.shock_embedding[index].float().cpu().tolist(),
                        "semantic_shift": output.semantic_shift[index].float().cpu().tolist(),
                        "cosine_shift": float(output.cosine_shift[index]),
                        "semantic_volatility": float(output.semantic_volatility[index]),
                        "shock_values": {
                            name: float(value)
                            for name, value in zip(model.shock_names, output.shock_values[index].float().cpu())
                        },
                    }
                )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


if __name__ == "__main__":
    main()

