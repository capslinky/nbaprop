"""Unit tests for feature pipeline helpers."""

from nbaprop.features.pipeline import build_features


def test_build_features_rest_days_and_opponent():
    props = [{
        "prop_id": "p1",
        "player_id": "player-1",
        "player_name": "Test Player",
        "prop_type": "points",
        "line": 10,
        "side": "over",
        "home_team": "AAA",
        "away_team": "BBB",
        "game_time": "2025-12-22T00:00:00Z",
        "team_abbrev": "AAA",
    }]
    raw_player_logs = [{
        "player": "Test Player",
        "logs": [
            {"points": 12, "minutes": 30, "date": "2025-12-20"},
            {"points": 8, "minutes": 28, "date": "2025-12-18"},
        ],
    }]
    players = [{"player_id": "player-1", "player_name": "Test Player", "team_abbrev": "AAA"}]

    features = build_features(props, raw_player_logs=raw_player_logs, players=players)
    payload = features[0]["features"]

    assert payload["rest_days"] == 2
    assert payload["opponent_team"] == "BBB"
    assert payload["is_home"] is True
