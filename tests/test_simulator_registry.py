from pathlib import Path

from dsge_rl.simulator_registry import select_simulators, simulator_registry


def test_registry_covers_every_simulator_file():
    registry = simulator_registry()
    registered_files = {Path(config.model_path).name for config in registry.values()}
    actual_files = {path.name for path in (Path(__file__).parent.parent / "DSGE_Simulator").glob("*.yaml")}
    assert registered_files == actual_files


def test_select_all_returns_every_profile():
    assert select_simulators("all").keys() == simulator_registry().keys()


def test_scenarios_have_discourse_for_every_turn():
    for config in simulator_registry().values():
        for scenario in config.scenarios:
            assert len(scenario.discourse) >= config.turns + 1

