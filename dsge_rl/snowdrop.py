from __future__ import annotations

import copy
import importlib
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class CommittedState:
    def __init__(self, period: int, values: dict[str, float], history: tuple[dict[str, float], ...]):
        self.period = period
        self.values = values
        self.history = history


class SnowdropUnavailableError(ImportError):
    pass


class SnowdropSimulator:
    def __init__(self, model_path: str | Path, periods: int = 40, solver: str | None = None):
        self.model_path = str(Path(model_path).resolve())
        self.periods = periods
        self.solver = solver
        try:
            driver = importlib.import_module("snowdrop.src.driver")
        except ImportError as error:
            raise SnowdropUnavailableError("Install Snowdrop and make snowdrop.src importable") from error
        self._run_model = driver.run
        with redirect_stdout(io.StringIO()):
            self.model = driver.importModel(fname=self.model_path)
        self.model.options["T"] = periods
        symbols = self.model.symbols
        self.variables = list(symbols.get("variables", []))
        self.shocks = list(symbols.get("shocks", []))
        self.parameters = list(symbols.get("parameters", []))
        self._initial_calibration = copy.deepcopy(self.model.calibration)
        self._initial_options = copy.deepcopy(self.model.options)
        self.committed_state = self.initial_state()

    def clone(self) -> "SnowdropSimulator":
        clone = object.__new__(SnowdropSimulator)
        clone.model_path = self.model_path
        clone.periods = self.periods
        clone.solver = self.solver
        clone._run_model = self._run_model
        clone.model = copy.deepcopy(self.model)
        clone.variables = self.variables.copy()
        clone.shocks = self.shocks.copy()
        clone.parameters = self.parameters.copy()
        clone._initial_calibration = copy.deepcopy(self._initial_calibration)
        clone._initial_options = copy.deepcopy(self._initial_options)
        clone.committed_state = CommittedState(
            self.committed_state.period,
            self.committed_state.values.copy(),
            tuple(item.copy() for item in self.committed_state.history),
        )
        return clone

    def initial_state(self) -> CommittedState:
        calibration = self.model.calibration.get("variables", np.zeros(len(self.variables)))
        values = {}
        for index, name in enumerate(self.variables):
            value = np.asarray(calibration[index])
            values[name] = float(value.reshape(-1)[-1]) if value.size else 0.0
        return CommittedState(0, values, (values.copy(),))

    def reset_transition(self) -> CommittedState:
        self.model.calibration = copy.deepcopy(self._initial_calibration)
        self.model.options = copy.deepcopy(self._initial_options)
        self.model.options["T"] = self.periods
        self.committed_state = self.initial_state()
        return self.committed_state

    def set_committed_state(self, state: CommittedState) -> None:
        calibration = np.asarray(self.model.calibration["variables"]).copy()
        for index, name in enumerate(self.variables):
            if name in state.values:
                if np.asarray(calibration[index]).ndim == 0:
                    calibration[index] = state.values[name]
                else:
                    calibration[index][...] = state.values[name]
        self.model.calibration["variables"] = calibration
        self.committed_state = state

    def forecast_from_state(
        self,
        state: CommittedState,
        shock_paths: dict[str, Any],
        horizon: int,
    ) -> pd.DataFrame:
        simulation = self.clone()
        simulation.periods = int(horizon)
        simulation.model.options["T"] = int(horizon)
        simulation.set_committed_state(state)
        simulation.inject_shocks(shock_paths)
        return simulation.run()

    def advance(
        self,
        shock_paths: dict[str, Any],
        horizon: int,
    ) -> tuple[CommittedState, pd.DataFrame]:
        forecast = self.forecast_from_state(self.committed_state, shock_paths, horizon)
        if forecast.empty:
            raise RuntimeError("Snowdrop returned an empty transition forecast")
        next_index = self._next_state_index(forecast, self.committed_state.values)
        row = forecast.iloc[next_index]
        values = {name: float(row[name]) for name in self.variables}
        history = (*self.committed_state.history, values.copy())
        state = CommittedState(self.committed_state.period + 1, values, history)
        self.set_committed_state(state)
        return state, forecast

    def _next_state_index(self, forecast: pd.DataFrame, current: dict[str, float]) -> int:
        if len(forecast) == 1:
            return 0
        comparable = [name for name in self.variables if name in forecast.columns and np.isfinite(current.get(name, np.nan))]
        if not comparable:
            return 0
        first = forecast.iloc[0][comparable].to_numpy(dtype=float)
        initial = np.array([current[name] for name in comparable], dtype=float)
        return 1 if np.allclose(first, initial, rtol=1e-5, atol=1e-7) else 0

    def inject_shocks(self, shock_paths: dict[str, Any]) -> None:
        matrix = np.zeros((self.periods, len(self.shocks)), dtype=float)
        for name, values in shock_paths.items():
            if name not in self.shocks:
                raise ValueError(f"Unknown shock {name}")
            column = self.shocks.index(name)
            if np.isscalar(values):
                matrix[0, column] = float(values)
            elif isinstance(values, pd.Series):
                vector = values.to_numpy(dtype=float)
                matrix[: min(len(vector), self.periods), column] = vector[: self.periods]
            else:
                vector = np.asarray(values, dtype=float)
                matrix[: min(len(vector), self.periods), column] = vector[: self.periods]
        self.model.options["shock_values"] = matrix
        if isinstance(getattr(self.model, "calibration", None), dict):
            self.model.calibration["shocks"] = matrix

    def update_parameters(self, values: dict[str, float]) -> None:
        calibration = self.model.calibration["parameters"]
        for name, value in values.items():
            if name not in self.parameters:
                raise ValueError(f"Unknown parameter {name}")
            calibration[self.parameters.index(name)] = value
        self.model.calibration["parameters"] = calibration

    def run(self, output_variables: tuple[str, ...] | None = None) -> pd.DataFrame:
        solver = self.solver or ("BinderPesaran" if self.model.options.get("linear", True) else "LBJ")
        with redirect_stdout(io.StringIO()):
            values, index = self._run_model(model=self.model, Solver=solver, Plot=False, Output=False)
        rows = min(len(index), len(values))
        frame = pd.DataFrame(values[:rows], index=index[:rows], columns=self.variables)
        if output_variables is None:
            return frame
        missing = set(output_variables).difference(frame.columns)
        if missing:
            raise ValueError(f"Unknown output variables: {sorted(missing)}")
        return frame.loc[:, list(output_variables)]

    def simulate(self, shock_paths: dict[str, Any], output_variables: tuple[str, ...] | None = None) -> pd.DataFrame:
        simulation = self.clone()
        simulation.inject_shocks(shock_paths)
        return simulation.run(output_variables)
