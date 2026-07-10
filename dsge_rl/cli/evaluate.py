import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsge_rl.config import load_config
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.rollout import collect_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(config.model.model_id, torch_dtype="auto", device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)
    environment = DSGEPolicyEnvironment(config.environment)
    results = []
    for scenario in config.environment.scenarios:
        trajectory = collect_trajectory(
            model,
            tokenizer,
            environment.clone(),
            scenario,
            config.training.max_new_tokens,
            config.training.temperature,
        )
        results.append({"scenario": scenario.name, "reward": trajectory[-1].reward, "actions": [turn.text for turn in trajectory]})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

