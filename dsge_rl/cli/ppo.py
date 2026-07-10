import argparse

from dsge_rl.config import load_config
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import load_policy, load_value_model
from dsge_rl.trainers.ppo import PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer = load_policy(config.model)
    value_model = load_value_model(config.model)
    environment = DSGEPolicyEnvironment(config.environment)
    PPOTrainer(model, value_model, tokenizer, environment, config).train()


if __name__ == "__main__":
    main()
