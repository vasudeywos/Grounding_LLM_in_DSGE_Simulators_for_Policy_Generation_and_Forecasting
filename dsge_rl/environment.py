from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from dsge_rl.config import EnvironmentConfig, ScenarioConfig
from dsge_rl.semantics import ModelSemantics, ParsedAction, parse_action
from dsge_rl.snowdrop import SnowdropSimulator


@dataclass(frozen=True)
class Transition:
    observation: dict[str, float]
    reward: float
    terminated: bool
    valid_action: bool
    economic_loss: float


class DSGEPolicyEnvironment:
    def __init__(self, config: EnvironmentConfig, simulator: SnowdropSimulator | None = None):
        self.config = config
        self.simulator = simulator or SnowdropSimulator(config.model_path, config.periods)
        self.semantics = ModelSemantics(config.model_path)
        self.target_names = tuple(target.variable for target in config.targets)
        self.scenario: ScenarioConfig | None = None
        self.turn = 0
        self.shock_paths: dict[str, list[float]] = {}
        self.baseline_loss = 0.0
        self.previous_loss = 0.0
        self.invalid_actions = 0
        self.action_energy = 0.0

    def reset(self, scenario: ScenarioConfig) -> dict[str, float]:
        self.scenario = scenario
        self.turn = 0
        self.invalid_actions = 0
        self.action_energy = 0.0
        self.shock_paths = {name: self._pad(values) for name, values in scenario.shocks.items()}
        baseline = self.simulator.simulate(self.shock_paths, self.target_names)
        self.baseline_loss = self._loss(baseline)
        self.previous_loss = self.baseline_loss
        return self._observation(baseline, self.config.burn_in)

    def step_text(self, text: str) -> Transition:
        return self.step(parse_action(text, self.config.levers))

    def step(self, action: ParsedAction) -> Transition:
        if self.scenario is None:
            raise RuntimeError("Call reset before step")
        action_energy = 0.0
        if action.valid:
            action_energy = self._apply(action.shocks)
        else:
            self.invalid_actions += 1
        self.turn += 1
        frame = self.simulator.simulate(self.shock_paths, self.target_names)
        economic_loss = self._loss(frame)
        terminated = self.turn >= self.config.turns
        reward = self._reward(economic_loss, action_energy, action.valid, terminated)
        self.previous_loss = economic_loss
        index = min(self.config.burn_in + self.turn, len(frame) - 1)
        return Transition(self._observation(frame, index), reward, terminated, action.valid, economic_loss)

    def system_prompt(self) -> str:
        lever_lines = [
            f"{lever.name}: shock={lever.shock}, range=[{lever.minimum}, {lever.maximum}]"
            for lever in self.config.levers
        ]
        context = self.semantics.context({lever.shock for lever in self.config.levers})
        return (
            "You are an economic policy agent grounded in a DSGE simulator. "
            "Choose one policy action per turn to minimize terminal economic loss. "
            "Return only JSON with schema {\"lever\": \"NAME\", \"magnitude\": number}.\n"
            + "\n".join(lever_lines)
            + "\n"
            + context
        )

    def user_prompt(self, observation: dict[str, float]) -> str:
        scenario = self.scenario.name if self.scenario else "unknown"
        return f"Scenario: {scenario}\nTurn: {self.turn + 1}/{self.config.turns}\nState: {json.dumps(observation, sort_keys=True)}"

    def _pad(self, values: list[float]) -> list[float]:
        return (list(values) + [0.0] * self.config.periods)[: self.config.periods]

    def _apply(self, shocks: dict[str, float]) -> float:
        start = self.config.burn_in + self.turn
        lever_by_shock = {lever.shock: lever for lever in self.config.levers}
        energy = 0.0
        for name, magnitude in shocks.items():
            path = self.shock_paths.setdefault(name, [0.0] * self.config.periods)
            lever = lever_by_shock[name]
            value = magnitude
            for offset in range(self.config.action_duration):
                index = start + offset
                if index < self.config.periods:
                    path[index] += value
                    value *= lever.persistence
            self.action_energy += magnitude * magnitude
            energy += magnitude * magnitude
        return energy

    def _loss(self, frame: pd.DataFrame) -> float:
        start = self.config.burn_in
        stop = len(frame)
        total = 0.0
        for target in self.config.targets:
            deviation = frame[target.variable].iloc[start:stop].to_numpy(dtype=float) - target.target
            total += target.weight * float(np.square(deviation).sum())
        return total

    def _reward(self, loss: float, action_energy: float, valid_action: bool, terminated: bool) -> float:
        improvement = (self.previous_loss - loss) / max(abs(self.baseline_loss), 1.0)
        volatility = self.scenario.semantic_volatility if self.scenario and terminated else 0.0
        penalty = self.config.volatility_weight * volatility + self.config.action_weight * action_energy
        if not valid_action:
            penalty += self.config.invalid_action_penalty
        return float(self.config.reward_scale * np.tanh(improvement) - penalty)

    def _observation(self, frame: pd.DataFrame, index: int) -> dict[str, float]:
        row = frame.iloc[index]
        return {target.label: round(float(row[target.variable]), 6) for target in self.config.targets}

    def clone(self) -> "DSGEPolicyEnvironment":
        clone = object.__new__(DSGEPolicyEnvironment)
        clone.config = self.config
        clone.simulator = self.simulator.clone()
        clone.semantics = self.semantics
        clone.target_names = self.target_names
        clone.scenario = self.scenario
        clone.turn = self.turn
        clone.shock_paths = copy.deepcopy(self.shock_paths)
        clone.baseline_loss = self.baseline_loss
        clone.previous_loss = self.previous_loss
        clone.invalid_actions = self.invalid_actions
        clone.action_energy = self.action_energy
        return clone
