from pathlib import Path

import pandas as pd

from dsge_rl.config import EnvironmentConfig, LeverConfig, ScenarioConfig, TargetConfig
from dsge_rl.environment import DSGEPolicyEnvironment


class FakeSimulator:
    def __init__(self):
        self.calls = 0

    def clone(self):
        return self

    def simulate(self, shocks, output_variables):
        self.calls += 1
        policy = sum(shocks.get("policy", []))
        return pd.DataFrame({"inflation": [2.0 - policy] * 12})


def test_each_turn_runs_simulator_and_returns_reward(tmp_path: Path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("symbols: {}")
    config = EnvironmentConfig(
        model_path=str(model_path),
        periods=12,
        burn_in=2,
        turns=2,
        action_duration=2,
        targets=(TargetConfig("inflation", "Inflation"),),
        levers=(LeverConfig("MONETARY", "policy", 0.0, 2.0, 1.0),),
        scenarios=(ScenarioConfig("INFLATION", {"cost": [1.0]}),),
    )
    simulator = FakeSimulator()
    environment = DSGEPolicyEnvironment(config, simulator)
    environment.reset(config.scenarios[0])
    first = environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    second = environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    assert simulator.calls == 3
    assert first.reward != 0.0
    assert second.terminated
