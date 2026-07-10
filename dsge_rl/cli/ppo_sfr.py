import argparse

from dsge_rl.config import ModelConfig, TrainingConfig
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.modeling import load_policy, load_value_model
from dsge_rl.sfr_conditioning import SFRConfig, SFRConditionedPolicy, SFRConditionedValueModel, SFRShiftEncoder
from dsge_rl.simulator_registry import select_simulators, validate_registry_files
from dsge_rl.trainers.sfr_ppo import SFRCrossSimulatorPPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulators", default="all")
    parser.add_argument("--output-dir", default="outputs/sfr_ppo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--sfr-4bit", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    validate_registry_files()
    model_config = ModelConfig()
    training = TrainingConfig(output_dir=args.output_dir, epochs=args.epochs)
    sfr_config = SFRConfig(load_in_4bit=args.sfr_4bit)
    semantic_encoder = SFRShiftEncoder(sfr_config)
    policy, tokenizer = load_policy(model_config)
    value_model = load_value_model(model_config)
    conditioned_policy = SFRConditionedPolicy(policy, semantic_encoder.dimension, sfr_config)
    conditioned_value = SFRConditionedValueModel(value_model, semantic_encoder.dimension, sfr_config)
    environments = {
        name: DSGEPolicyEnvironment(config)
        for name, config in select_simulators(args.simulators, args.periods, args.turns).items()
    }
    SFRCrossSimulatorPPOTrainer(
        conditioned_policy,
        conditioned_value,
        tokenizer,
        semantic_encoder,
        environments,
        training,
    ).train()


if __name__ == "__main__":
    main()

