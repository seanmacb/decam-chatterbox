"""Tests for decam_chatterbox.heartbeat."""

import json
import time

import pytest
from astropy.time import Time, TimeDelta

from decam_chatterbox.heartbeat import Heartbeat, ServiceStatus, post_shutdown, post_startup
from decam_chatterbox.slackbot.client import SlackPoster


def _status(**overrides) -> ServiceStatus:
    defaults = dict(started_at=Time.now() - TimeDelta(3661, format="sec"), origin="scimma test")
    defaults.update(overrides)
    return ServiceStatus(**defaults)


def test_service_status_uptime_is_about_the_elapsed_time():
    status = _status()
    assert status.uptime_s == pytest.approx(3661, abs=2)


def test_service_status_starts_with_no_handled_alerts():
    status = _status()
    assert status.handled == 0
    assert status.last_handled_at is None


def test_heartbeat_disabled_never_starts_a_thread(config):
    config.heartbeat.enabled = False
    heartbeat = Heartbeat(config, SlackPoster(config), _status())
    heartbeat.start()
    assert heartbeat._thread is None
    heartbeat.stop()  # must not raise or hang when nothing was started


def test_heartbeat_stop_without_start_does_not_raise(config):
    heartbeat = Heartbeat(config, SlackPoster(config), _status())
    heartbeat.stop()


def test_heartbeat_posts_on_its_configured_interval(config):
    config.heartbeat.enabled = True
    config.heartbeat.interval_s = 0.05
    poster = SlackPoster(config)
    status = _status(handled=3)
    heartbeat = Heartbeat(config, poster, status)

    heartbeat.start()
    try:
        deadline = time.monotonic() + 2.0
        payload_path = poster.output_dir / "heartbeat.json"
        while time.monotonic() < deadline and not payload_path.is_file():
            time.sleep(0.02)
        assert payload_path.is_file(), "heartbeat did not post within 2 s of a 0.05 s interval"
    finally:
        heartbeat.stop()
    assert not heartbeat._thread.is_alive()

    payload = json.loads(payload_path.read_text())
    assert "heartbeat" in payload["text"].lower()
    assert "3 alert(s) handled" in payload["text"]
    assert "scimma test" in payload["text"]


def test_heartbeat_uses_the_configured_channel(config):
    config.heartbeat.interval_s = 0.05
    config.slack.channel = "#main"
    config.slack.heartbeat_channel = "#status"
    poster = SlackPoster(config)
    heartbeat = Heartbeat(config, poster, _status())

    heartbeat.start()
    try:
        deadline = time.monotonic() + 2.0
        payload_path = poster.output_dir / "heartbeat.json"
        while time.monotonic() < deadline and not payload_path.is_file():
            time.sleep(0.02)
    finally:
        heartbeat.stop()

    payload = json.loads(payload_path.read_text())
    assert payload["channel"] == "#status"


def test_post_shutdown_writes_an_offline_payload(config):
    poster = SlackPoster(config)
    post_shutdown(_status(handled=5), config, poster, reason="interrupted")

    payload = json.loads((poster.output_dir / "shutdown.json").read_text())
    assert "shutting down" in payload["text"].lower()
    assert "interrupted" in payload["text"].lower()
    assert "5 alert(s) handled" in payload["text"]


def test_post_shutdown_with_no_poster_does_not_raise(config):
    post_shutdown(_status(), config, None)


class _ChannelNotFoundError(Exception):
    """Shaped like slack_sdk's SlackApiError: the code is on `.response`."""

    def __init__(self):
        super().__init__("The request to the Slack API failed.")
        self.response = {"ok": False, "error": "channel_not_found"}


class _RecordingPoster:
    """Fails for chosen channels, records every post."""

    def __init__(self, failing=()):
        self.failing = set(failing)
        self.posts = []

    def post(self, blocks, text, channel=None, **kwargs):
        self.posts.append({"channel": channel, "blocks": blocks, "text": text, **kwargs})
        if channel in self.failing:
            raise _ChannelNotFoundError()


def test_status_post_falls_back_to_the_main_channel_with_the_reason(config):
    config.slack.channel = "#main"
    config.slack.heartbeat_channel = "#private-status"
    poster = _RecordingPoster(failing={"#private-status"})

    Heartbeat(config, poster, _status())._post()

    assert [p["channel"] for p in poster.posts] == ["#private-status", None]
    fallback = poster.posts[-1]
    assert "channel_not_found" in fallback["text"]
    assert "#private-status" in fallback["blocks"][-1]["elements"][0]["text"]
    assert "invite" in fallback["blocks"][-1]["elements"][0]["text"]


def test_status_post_does_not_retry_when_there_is_no_separate_channel(config):
    config.slack.heartbeat_channel = ""
    poster = _RecordingPoster(failing={None})
    with pytest.raises(_ChannelNotFoundError):
        Heartbeat(config, poster, _status())._post()
    assert len(poster.posts) == 1


def test_shutdown_notice_also_falls_back_to_the_main_channel(config):
    config.slack.channel = "#main"
    config.slack.heartbeat_channel = "#private-status"
    poster = _RecordingPoster(failing={"#private-status"})

    post_shutdown(_status(), config, poster)

    assert len(poster.posts) == 2
    assert "shutting down" in poster.posts[-1]["text"].lower()


def test_post_startup_reports_the_origin_and_never_raises(config):
    poster = SlackPoster(config)
    post_startup(_status(origin="scimma test"), config, poster)
    payload = json.loads((poster.output_dir / "startup.json").read_text())
    assert "started" in payload["text"]
    assert "scimma test" in payload["text"]

    config.slack.heartbeat_channel = ""
    post_startup(_status(), config, _RecordingPoster(failing={None}))  # both fail: still no raise
    post_startup(_status(), config, None)
