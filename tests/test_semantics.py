from dsge_rl.config import LeverConfig
from dsge_rl.semantics import parse_action


LEVERS = (LeverConfig("MONETARY", "SHK_RS", -2.0, 2.0),)


def test_valid_action():
    action = parse_action('{"lever":"MONETARY","magnitude":0.5}', LEVERS)
    assert action.valid
    assert action.shocks == {"SHK_RS": 0.5}


def test_rejects_out_of_range_action():
    action = parse_action('{"lever":"MONETARY","magnitude":4}', LEVERS)
    assert not action.valid


def test_rejects_non_json_action():
    action = parse_action("raise rates", LEVERS)
    assert not action.valid

