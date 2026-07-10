from __future__ import annotations

from pathlib import Path

from dsge_rl.config import EnvironmentConfig, LeverConfig, ScenarioConfig, TargetConfig


SIMULATOR_DIRECTORY = Path(__file__).resolve().parent.parent / "DSGE_Simulator"


def _path(name: str) -> str:
    return str((SIMULATOR_DIRECTORY / name).resolve())


def _discourse(topic: str, reversal: str) -> tuple[str, ...]:
    return (
        f"Economic discussion is broadly stable before news about {topic}.",
        f"New reports indicate {topic}, changing expectations across markets.",
        f"Analysts and households increasingly discuss the consequences of {topic}.",
        f"Policy debate intensifies as evidence about {topic} accumulates.",
        f"Later reporting suggests {reversal} and expectations begin to stabilize.",
    )


def simulator_registry(periods: int = 40, turns: int = 4) -> dict[str, EnvironmentConfig]:
    shared = {
        "periods": periods,
        "burn_in": 8,
        "turns": turns,
        "action_duration": 4,
        "mode": "control",
        "forecast_horizon": 8,
        "volatility_weight": 0.2,
        "action_weight": 0.01,
        "invalid_action_penalty": 2.0,
        "reward_scale": 10.0,
    }
    return {
        "qpm": EnvironmentConfig(
            model_path=_path("model.yaml"),
            targets=(TargetConfig("DLA_CPI", "Inflation", 0.0, 2.0), TargetConfig("L_GDP_GAP", "Output Gap", 0.0, 1.0)),
            levers=(LeverConfig("MONETARY", "SHK_RS", -2.0, 2.0), LeverConfig("FISCAL", "SHK_L_GDP_GAP", -2.0, 2.0)),
            scenarios=(
                ScenarioConfig("STAGFLATION", {"SHK_DLA_CPI": [2.0] * 4, "SHK_L_GDP_GAP": [-1.0] * 4}, 0.7, _discourse("persistent inflation and weakening output", "price pressures are easing")),
                ScenarioConfig("RECESSION", {"SHK_L_GDP_GAP": [-1.5] * 4}, 0.5, _discourse("a sharp contraction in aggregate demand", "demand is recovering")),
            ),
            **shared,
        ),
        "sw": EnvironmentConfig(
            model_path=_path("sw_model.yaml"),
            targets=(TargetConfig("pinf", "Inflation", 0.0, 2.0), TargetConfig("y", "Output", 0.0, 1.0)),
            levers=(LeverConfig("MONETARY", "em", -0.5, 0.5), LeverConfig("FISCAL", "eg", -0.5, 0.5)),
            scenarios=(
                ScenarioConfig("COST_PUSH", {"epinf": [0.5] * 4}, 0.6, _discourse("a broad price-markup shock", "input costs are normalizing")),
                ScenarioConfig("PRODUCTIVITY", {"ea": [-0.5] * 4}, 0.6, _discourse("an unexpected productivity decline", "productivity is recovering")),
            ),
            **shared,
        ),
        "gsw": EnvironmentConfig(
            model_path=_path("gsw_model.yaml"),
            targets=(TargetConfig("pinf", "Inflation", 0.0, 1.5), TargetConfig("y", "Output", 0.0, 1.0), TargetConfig("unempl", "Unemployment", 0.0, 1.0)),
            levers=(LeverConfig("MONETARY", "em", -0.5, 0.5), LeverConfig("FISCAL", "eg", -0.5, 0.5)),
            scenarios=(
                ScenarioConfig("LOCKDOWN", {"els": [-0.5] * 4, "ey": [-0.5] * 4}, 0.9, _discourse("lockdown restrictions and a collapse in labor activity", "economic reopening is progressing")),
                ScenarioConfig("PRICE_MARKUP", {"epinf": [0.5] * 4}, 0.6, _discourse("firms raising markups amid supply constraints", "markup pressures are fading")),
            ),
            **shared,
        ),
        "ireland": EnvironmentConfig(
            model_path=_path("Ireland2004.yaml"),
            targets=(TargetConfig("pie", "Inflation", 0.0, 2.0), TargetConfig("x", "Output Gap", 0.0, 1.0)),
            levers=(LeverConfig("MONETARY", "epsr", -0.05, 0.05),),
            scenarios=(
                ScenarioConfig("COST_PUSH", {"epse": [0.01] * 4}, 0.5, _discourse("an inflationary cost disturbance", "cost pressures are receding")),
                ScenarioConfig("TECHNOLOGY", {"epsa": [-0.05] * 4}, 0.5, _discourse("a negative technology shock", "technology conditions are improving")),
            ),
            **shared,
        ),
        "mvf_us": EnvironmentConfig(
            model_path=_path("MVF_US.yaml"),
            targets=(TargetConfig("PIE", "Inflation", 2.0, 2.0), TargetConfig("Y", "Output Gap", 0.0, 1.0), TargetConfig("UNR_GAP", "Unemployment Gap", 0.0, 1.0)),
            levers=(LeverConfig("DEMAND_STABILIZER", "RES_Y", -1.0, 1.0),),
            scenarios=(
                ScenarioConfig("INFLATION", {"RES_PIE": [0.5] * 4}, 0.5, _discourse("unexpected inflation pressure", "inflation expectations are stabilizing")),
                ScenarioConfig("POTENTIAL_OUTPUT", {"RES_LGDP_BAR": [-0.5] * 4}, 0.6, _discourse("a decline in potential output", "productive capacity is recovering")),
            ),
            **shared,
        ),
        "rbc": EnvironmentConfig(
            model_path=_path("RBC.yaml"),
            targets=(TargetConfig("Y", "Output", 1.2, 1.0), TargetConfig("C", "Consumption", 0.8, 1.0)),
            levers=(LeverConfig("OUTPUT_STABILIZER", "ey", -0.1, 0.1),),
            scenarios=(
                ScenarioConfig("TECHNOLOGY", {"ea": [-0.05] * 4}, 0.5, _discourse("a negative productivity innovation", "productivity is reverting toward trend")),
            ),
            **shared,
        ),
    }


def select_simulators(names: str | None, periods: int = 40, turns: int = 4) -> dict[str, EnvironmentConfig]:
    registry = simulator_registry(periods, turns)
    if names is None or names.lower() == "all":
        return registry
    selected = [name.strip().lower() for name in names.split(",") if name.strip()]
    unknown = sorted(set(selected).difference(registry))
    if unknown:
        raise ValueError(f"Unknown simulators: {unknown}. Available: {sorted(registry)}")
    return {name: registry[name] for name in selected}


def validate_registry_files() -> None:
    missing = [config.model_path for config in simulator_registry().values() if not Path(config.model_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing DSGE simulator files: {missing}")
