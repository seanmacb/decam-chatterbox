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


def _ping_config(tmp_path, mention='["!subteam^S123"]'):
    path = tmp_path / "config.yaml"
    path.write_text(
        f"paths:\n  work_dir: {tmp_path / 'work'}\n"
        f"slack:\n  channel: '#main'\n  urgent_channel: '#urgent'\n  mention: {mention}\n"
    )
    return path


def test_test_ping_refuses_when_nobody_is_configured_to_ping(tmp_path, capsys):
    rc = main(["-c", str(_ping_config(tmp_path, mention="[]")), "test-ping", "--yes"])
    assert rc == 2
    assert "slack.mention is empty" in capsys.readouterr().err


def test_test_ping_offline_writes_a_payload_that_mentions_the_team(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    rc = main(["-c", str(_ping_config(tmp_path)), "test-ping"])
    assert rc == 1  # offline: nothing was actually sent
    assert "Offline" in capsys.readouterr().out
    payload = json.loads((tmp_path / "work" / "posts" / "test_ping.json").read_text())
    assert payload["channel"] == "#urgent"
    assert "!subteam^S123" in payload["text"]
    assert "!subteam^S123" in payload["blocks"][0]["text"]["text"]
    assert payload["blocks"][0]["text"]["text"].startswith(":bell: *On-call:*")
    assert "TEST" in payload["blocks"][2]["text"]["text"]


def test_test_ping_asks_before_notifying_and_can_be_cancelled(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    rc = main(["-c", str(_ping_config(tmp_path)), "test-ping"])
    assert rc == 1
    assert "Cancelled" in capsys.readouterr().out


def test_test_ping_override_mention_and_channel(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    sent = {}

    def fake_post(self, blocks, text, **kwargs):
        from decam_chatterbox.slackbot.client import PostedMessage

        sent.update(kwargs, text=text, mention_list=list(self.config.slack.mention))
        return PostedMessage(channel=kwargs["channel"], ts="1.0")

    monkeypatch.setattr("decam_chatterbox.slackbot.client.SlackPoster.post", fake_post)
    rc = main(
        [
            "-c",
            str(_ping_config(tmp_path)),
            "test-ping",
            "--yes",
            "--mention",
            "U0123ABCD",
            "--channel",
            "#me",
        ]
    )
    assert rc == 0
    assert sent["channel"] == "#me"
    assert sent["mention"] is True
    assert sent["mention_list"] == ["@U0123ABCD"]


def test_test_ping_normalizes_a_pasted_group_mention(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    main(["-c", str(_ping_config(tmp_path)), "test-ping", "--mention", "@decam-ir1-team^S0C4J7UE4NB"])
    payload = json.loads((tmp_path / "work" / "posts" / "test_ping.json").read_text())
    assert payload["text"].startswith("<!subteam^S0C4J7UE4NB> ")


def test_test_ping_rejects_an_unresolvable_handle(tmp_path, capsys):
    rc = main(["-c", str(_ping_config(tmp_path)), "test-ping", "--yes", "--mention", "@some-team"])
    assert rc == 2
    assert "Cannot make a Slack mention" in capsys.readouterr().err
