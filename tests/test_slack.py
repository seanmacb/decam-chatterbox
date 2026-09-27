"""Tests for decam_chatterbox.slackbot.client."""

import json

from decam_chatterbox.slackbot.client import SlackPoster, render_blocks_as_text


def test_poster_is_offline_without_a_token(config):
    poster = SlackPoster(config)
    assert poster.offline


def test_offline_post_writes_payload_to_disk(config):
    poster = SlackPoster(config)
    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
    posted = poster.post(blocks, "hello", label="mytest")
    assert posted.offline
    assert posted.ts is None

    written = json.loads((poster.output_dir / "mytest.json").read_text())
    assert written["text"] == "hello"
    assert written["blocks"] == blocks


def test_channel_for_routes_test_alerts_to_test_channel(config):
    config.slack.channel = "#main"
    config.slack.test_channel = "#test"
    poster = SlackPoster(config)
    assert poster.channel_for(is_test=True) == "#test"
    assert poster.channel_for(is_test=False) == "#main"


def test_channel_for_falls_back_to_main_when_no_test_channel(config):
    config.slack.channel = "#main"
    config.slack.test_channel = ""
    poster = SlackPoster(config)
    assert poster.channel_for(is_test=True) == "#main"


def test_upload_missing_file_returns_false(config):
    poster = SlackPoster(config)
    assert poster.upload("/does/not/exist.png", channel="#main") is False


def test_dry_run_forces_offline_even_with_a_token(config, monkeypatch):
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    poster = SlackPoster(config, dry_run=True)
    assert poster.offline


def test_render_blocks_as_text_covers_every_block_type():
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": "Title"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "body"}},
        {"type": "divider"},
        {"type": "context", "elements": [{"type": "mrkdwn", "text": "footer"}]},
    ]
    text = render_blocks_as_text(blocks)
    assert "Title" in text
    assert "body" in text
    assert "footer" in text
    assert "-" * 10 in text
