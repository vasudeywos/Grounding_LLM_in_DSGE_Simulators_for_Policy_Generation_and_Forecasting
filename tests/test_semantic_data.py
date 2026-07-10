import json

from semantic_shift.data import build_pairs, chronological_split, load_windows


def test_pairs_remain_within_source_and_time_order(tmp_path):
    rows = [
        {"timestamp": "2024-01-02", "source": "a", "texts": ["second"]},
        {"timestamp": "2024-01-01", "source": "b", "texts": ["other"]},
        {"timestamp": "2024-01-01", "source": "a", "texts": ["first"]},
        {"timestamp": "2024-01-03", "source": "a", "texts": ["third"]},
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    pairs = build_pairs(load_windows(path))
    assert len(pairs) == 2
    assert pairs[0][0].timestamp == "2024-01-01"
    assert all(previous.source == current.source for previous, current in pairs)


def test_chronological_split_has_no_future_leakage():
    pairs = [(index, index + 1) for index in range(10)]
    train, validation = chronological_split(pairs, 0.2)
    assert train[-1][1] <= validation[0][0]

