from dsge_rl.pandemic_validation import GSWPandemicMapper, PandemicDataset


def test_loads_published_usa_shocks():
    values, approximate = PandemicDataset().load_shocks("S01", "USA")
    assert not approximate
    assert "SHKC" in values
    assert "GOVS" in values


def test_scenario_six_is_explicitly_approximate():
    values, approximate = PandemicDataset().load_shocks("S06", "USA")
    assert approximate
    assert values


def test_mapper_creates_gsw_shock_paths():
    source = {
        "TFPP01": -4.0,
        "TFPP02": -2.0,
        "SHKC": -5.0,
        "RISH": 2.0,
        "EXCR": 1.0,
        "RISEP01": 3.0,
        "GOVS": 4.0,
        "MORB": -0.1,
    }
    paths = GSWPandemicMapper(periods=12).map(source)
    assert set(paths) == {"ea", "ey", "eb", "eg", "els"}
    assert all(len(path) == 12 for path in paths.values())

