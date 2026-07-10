import argparse

from semantic_shift.config import load_semantic_config
from semantic_shift.trainer import SemanticShiftTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_semantic_config(args.config)
    SemanticShiftTrainer(config).train()


if __name__ == "__main__":
    main()

