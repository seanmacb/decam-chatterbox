"""Tests for decam_chatterbox.config."""

import pytest

from decam_chatterbox.config import Config, load_config


def test_default_config_has_sensible_scimma_defaults():
    cfg = Config()
    assert cfg.ingest.kind == "scimma"
    assert cfg.ingest.hop_url == "kafka://kafka.scimma.org/igwn.gwalert"
    assert cfg.site == "ctio"


def test_load_config_missing_explicit_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does-not-exist.yaml")


def test_load_config_no_file_anywhere_returns_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    assert cfg == Config()


def test_load_config_reads_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("""
slack:
  channel: "#my-channel"
ingest:
  kind: files
  watch_dir: /tmp/incoming
dark_hours:
  nside: 128
""")
    cfg = load_config(path)
    assert cfg.slack.channel == "#my-channel"
    assert cfg.ingest.kind == "files"
    assert cfg.ingest.watch_dir == "/tmp/incoming"
    assert cfg.dark_hours.nside == 128
    # Unset sections keep their dataclass defaults.
    assert cfg.enrich.gracedb is True


def test_load_config_warns_on_unknown_key(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("slack:\n  channell: '#typo'\n")
    with caplog.at_level("WARNING"):
        cfg = load_config(path)
    assert cfg.slack.channel == "#decam-gw-alerts"  # the dataclass default, unaffected
    assert any("channell" in record.message for record in caplog.records)


def test_slack_token_reads_configured_env_var(monkeypatch):
    cfg = Config()
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    assert cfg.slack_token == "xoxb-test"


def test_slack_token_none_when_unset(monkeypatch):
    cfg = Config()
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    assert cfg.slack_token is None
