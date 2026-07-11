from gamescope_agent.models import GameRecord


def test_game_record_formats_search_document() -> None:
    record = GameRecord.model_validate(
        {
            "Name": "Example Game",
            "Platform": "Example Console",
            "Genre": "Adventure",
            "Publisher": "Example Publisher",
            "Description": "A compact test record.",
            "YearOfRelease": 2020,
            "SourceFile": "001.json",
        }
    )

    document = record.as_document()
    assert "Game title: Example Game" in document
    assert "Year of release: 2020" in document
