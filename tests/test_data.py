from pathlib import Path

from gamescope_agent.data import load_game_records


def test_sample_dataset_is_valid() -> None:
    data_dir = Path(__file__).parents[1] / "data" / "games"
    records = load_game_records(data_dir)

    assert len(records) == 15
    assert len({record.name for record in records}) == 15
    assert all(record.source_file for record in records)
