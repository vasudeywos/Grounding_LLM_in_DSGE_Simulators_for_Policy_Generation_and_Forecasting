import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from dsge_rl.config import ModelConfig
from dsge_rl.pandemic_validation import PandemicValidator, write_validation_report
from dsge_rl.rollout import collect_sfr_trajectory
from dsge_rl.sfr_conditioning import SFRConfig, SFRConditionedPolicy, SFRShiftEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--region", default="USA")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="outputs/pandemic_validation.json")
    parser.add_argument("--percentage-scale", type=float, default=0.01)
    args = parser.parse_args()
    scenarios = [f"S{index:02d}" for index in range(1, 7)] if args.scenario.lower() == "all" else [args.scenario.upper()]
    validator = PandemicValidator(percentage_scale=args.percentage_scale)
    model = None
    tokenizer = None
    semantic_encoder = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        semantic_encoder = SFRShiftEncoder(SFRConfig())
        model = SFRConditionedPolicy.from_pretrained(checkpoint / "policy", semantic_encoder.dimension, ModelConfig())
        tokenizer = AutoTokenizer.from_pretrained(checkpoint / "policy" / "tokenizer")
    reports = []
    for scenario in scenarios:
        try:
            report = validator.validate_simulator(scenario, args.region)
            if model is not None:
                environment = validator.build_policy_environment(scenario, args.region)
                task = environment.config.scenarios[0]
                trajectory = collect_sfr_trajectory(model, tokenizer, semantic_encoder, environment, task, 80, 1.0)
                report["policy"] = {
                    "trajectory_return": sum(turn.reward for turn in trajectory),
                    "turn_rewards": [turn.reward for turn in trajectory],
                    "no_policy_loss": environment.baseline_loss,
                    "policy_loss": environment.previous_loss,
                    "loss_improvement": environment.baseline_loss - environment.previous_loss,
                    "actions": [turn.text for turn in trajectory],
                }
            reports.append(report)
        except Exception as error:
            reports.append({"scenario": scenario, "region": args.region.upper(), "error": str(error)})
    result = {"dataset_doi": "10.17632/4d7rfm4s77.1", "reports": reports}
    write_validation_report(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
