from pathlib import Path

import pandas as pd

from dsge_rl.config import EnvironmentConfig, LeverConfig, ScenarioConfig, TargetConfig
from dsge_rl.environment import DSGEPolicyEnvironment
from dsge_rl.snowdrop import CommittedState


class FakeSimulator:
    def __init__(self):
        self.calls = 0
        self.committed_state = CommittedState(0, {"inflation": 2.0}, ({"inflation": 2.0},))

    def clone(self):
        return self

    def simulate(self, shocks, output_variables):
        self.calls += 1
        policy = sum(shocks.get("policy", []))
        return pd.DataFrame({"inflation": [2.0 - policy] * 12})

    def reset_transition(self):
        self.committed_state = CommittedState(0, {"inflation": 2.0}, ({"inflation": 2.0},))
        return self.committed_state

    def advance(self, shocks, horizon):
        self.calls += 1
        policy = shocks.get("policy", [0.0])[0]
        inflation = self.committed_state.values["inflation"] - policy
        values = {"inflation": inflation}
        self.committed_state = CommittedState(
            self.committed_state.period + 1,
            values,
            (*self.committed_state.history, values.copy()),
        )
        return self.committed_state, pd.DataFrame({"inflation": [inflation] * horizon})


def test_each_turn_runs_simulator_and_returns_reward(tmp_path: Path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("symbols: {}")
    config = EnvironmentConfig(
        model_path=str(model_path),
        periods=12,
        burn_in=0,
        turns=2,
        action_duration=2,
        targets=(TargetConfig("inflation", "Inflation"),),
        levers=(LeverConfig("MONETARY", "policy", 0.0, 2.0, 1.0),),
        scenarios=(ScenarioConfig("INFLATION", {"cost": [0.0]}),),
    )
    simulator = FakeSimulator()
    environment = DSGEPolicyEnvironment(config, simulator)
    environment.reset(config.scenarios[0])
    first = environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    second = environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    assert simulator.calls == 2
    assert first.reward != 0.0
    assert second.terminated


def test_committed_history_is_not_rewritten(tmp_path: Path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("symbols: {}")
    config = EnvironmentConfig(
        model_path=str(model_path),
        periods=8,
        burn_in=0,
        turns=2,
        action_duration=1,
        targets=(TargetConfig("inflation", "Inflation"),),
        levers=(LeverConfig("MONETARY", "policy", -2.0, 2.0, 1.0),),
        scenarios=(ScenarioConfig("INFLATION", {}),),
    )
    first_environment = DSGEPolicyEnvironment(config, FakeSimulator())
    second_environment = DSGEPolicyEnvironment(config, FakeSimulator())
    first_environment.reset(config.scenarios[0])
    second_environment.reset(config.scenarios[0])
    first_environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    second_environment.step_text('{"lever":"MONETARY","magnitude":0.5}')
    first_committed = first_environment.simulator.committed_state.history[1].copy()
    second_committed = second_environment.simulator.committed_state.history[1].copy()
    first_environment.step_text('{"lever":"MONETARY","magnitude":1.0}')
    second_environment.step_text('{"lever":"MONETARY","magnitude":-1.0}')
    assert first_committed == second_committed
    assert first_environment.simulator.committed_state.history[1] == first_committed
    assert second_environment.simulator.committed_state.history[1] == second_committed
    assert first_environment.simulator.committed_state.values != second_environment.simulator.committed_state.values
