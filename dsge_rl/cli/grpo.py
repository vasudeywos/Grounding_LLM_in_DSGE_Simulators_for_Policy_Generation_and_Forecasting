import argparse

from dsge_rl.config import load_config
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import load_policy
from dsge_rl.trainers.grpo import GRPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer = load_policy(config.model)
    environment = DSGEPolicyEnvironment(config.environment)
    GRPOTrainer(model, tokenizer, environment, config).train()


if __name__ == "__main__":
    main()

