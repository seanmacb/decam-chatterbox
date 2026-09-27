"""Tests for decam_chatterbox.cli."""

import json

from decam_chatterbox.cli import main


def _write_config(tmp_path, work_dir):
    path = tmp_path / "config.yaml"
    path.write_text(f"paths:\n  work_dir: {work_dir}\nenrich:\n  gracedb: false\n")
    return path


def test_replay_dry_run(tmp_path, notice, capsys):
    config_path = _write_config(tmp_path, tmp_path / "work")
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(notice))

    rc = main(["-c", str(config_path), "replay", str(record_path), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert notice["superevent_id"] in out
    assert "Gravitational wave" in out  # the rendered Block Kit message text
    assert "plot:" in out


def test_replay_reports_failure_on_bad_record(tmp_path, capsys):
    config_path = _write_config(tmp_path, tmp_path / "work")
    record_path = tmp_path / "bad.json"
    record_path.write_text(json.dumps({"alert_type": "PRELIMINARY"}))  # missing required fields

    rc = main(["-c", str(config_path), "replay", str(record_path), "--dry-run"])
    assert rc == 1


def test_test_post_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    config_path = _write_config(tmp_path, tmp_path / "work")
    rc = main(["-c", str(config_path), "test-post"])
    assert rc == 1  # offline: nothing was actually sent
    assert "Offline" in capsys.readouterr().out


def test_doctor_runs_and_prints_a_report(tmp_path, capsys):
    config_path = _write_config(tmp_path, tmp_path / "work")
    main(["-c", str(config_path), "doctor"])
    out = capsys.readouterr().out
    assert "decam-chatterbox environment" in out
    assert "interpreter" in out


def test_missing_config_path_is_a_clean_error(tmp_path, capsys):
    rc = main(["-c", str(tmp_path / "nope.yaml"), "doctor"])
    assert rc == 2
    assert "Could not load configuration" in capsys.readouterr().err
