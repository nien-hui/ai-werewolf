import pytest

from werewolf import config


@pytest.fixture(autouse=True)
def game_defaults(monkeypatch, tmp_path):
    """Test defaults, whatever .env says: games don't wait after AI speeches (test_graph checks the
    pause on its own), no password (test_password sets its own), and finished games are saved
    under the test's tmp_path instead of the project's games/."""
    monkeypatch.setattr(config, "SPEECH_PAUSE_S", 0.0)
    monkeypatch.setattr(config, "TYPE_S_PER_CHAR", 0.0)
    monkeypatch.setattr(config, "PASSWORD", "")
    monkeypatch.setattr(config, "GAMES_DIR", tmp_path / "games")
