from __future__ import annotations

import copy
import importlib
import io
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
        return clone

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

