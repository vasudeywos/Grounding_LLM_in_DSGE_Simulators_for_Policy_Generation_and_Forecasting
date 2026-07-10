import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from dsge_rl.config import ModelConfig
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.rollout import collect_sfr_trajectory
from dsge_rl.sfr_conditioning import SFRConfig, SFRConditionedPolicy, SFRShiftEncoder
from dsge_rl.simulator_registry import select_simulators, validate_registry_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--simulators", default="all")
    parser.add_argument("--output", default="outputs/cross_simulator_results.json")
    parser.add_argument("--periods", type=int, default=40)
    parser.add_argument("--turns", type=int, default=4)
    args = parser.parse_args()
    validate_registry_files()
    checkpoint = Path(args.checkpoint)
    semantic_encoder = SFRShiftEncoder(SFRConfig())
    model = SFRConditionedPolicy.from_pretrained(checkpoint / "policy", semantic_encoder.dimension, ModelConfig())
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "policy" / "tokenizer")
    results = []
    for simulator_name, config in select_simulators(args.simulators, args.periods, args.turns).items():
        for scenario in config.scenarios:
            try:
                environment = DSGEPolicyEnvironment(config)
                trajectory = collect_sfr_trajectory(
                    model,
                    tokenizer,
                    semantic_encoder,
                    environment,
                    scenario,
                    80,
                    1.0,
                )
                results.append(
                    {
                        "simulator": simulator_name,
                        "scenario": scenario.name,
                        "status": "passed",
                        "trajectory_return": sum(turn.reward for turn in trajectory),
                        "turn_rewards": [turn.reward for turn in trajectory],
                        "actions": [turn.text for turn in trajectory],
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "simulator": simulator_name,
                        "scenario": scenario.name,
                        "status": "failed",
                        "error": str(error),
                    }
                )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

