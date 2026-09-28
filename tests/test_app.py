"""End-to-end tests for decam_chatterbox.app, offline throughout."""

import json
import signal as signal_module

from conftest import make_notice

import decam_chatterbox.app as app_module
from decam_chatterbox.app import TriggerReport, post_failure, process_notice, run_service
from decam_chatterbox.slackbot.client import SlackPoster


def test_process_notice_end_to_end(notice, config):
    report = process_notice(notice, config)
    assert isinstance(report, TriggerReport)
    assert report.trigger.superevent_id == "S260814a"
    assert report.events is not None
    assert report.dark_hours is not None
    assert report.dark_stats is not None
    assert report.priority.is_high_priority  # conftest.make_notice's defaults qualify
    assert len(report.plots) == 1
    assert report.plots[0].is_file()
    assert report.posted is not None
    assert report.posted.offline  # no SLACK_BOT_TOKEN in the test environment
    assert report.posted_urgent is None  # no urgent_channel configured
    assert not report.warnings


def test_process_notice_cross_posts_high_priority_to_urgent_channel(notice, config):
    config.slack.urgent_channel = "#decam-urgent"
    report = process_notice(notice, config)
    assert report.priority.is_high_priority
    assert report.posted_urgent is not None
    assert report.posted_urgent.channel == "#decam-urgent"


def test_process_notice_does_not_cross_post_low_priority(notice, config):
    config.slack.urgent_channel = "#decam-urgent"
    config.priority.max_far_hz = 0.0  # nothing can pass this
    report = process_notice(notice, config)
    assert not report.priority.is_high_priority
    assert report.posted_urgent is None
    assert report.posted is not None  # still posted to the main channel


def test_process_notice_retraction_skips_observability(retraction_notice, config):
    report = process_notice(retraction_notice, config)
    assert report.events is None
    assert report.dark_hours is None
    assert report.plots == []


def test_process_notice_post_false_does_not_write_offline_payload(notice, config):
    report = process_notice(notice, config, post=False)
    assert report.posted is None
    payload = config_work_posts_dir(config) / f"{report.trigger.superevent_id}_preliminary.json"
    assert not payload.exists()


def test_process_notice_reuses_a_shared_poster(notice, config):
    poster = SlackPoster(config)
    report1 = process_notice(notice, config, poster=poster)
    report2 = process_notice(make_notice(superevent_id="S260814b"), config, poster=poster)
    assert report1.posted.offline and report2.posted.offline


def test_post_failure_with_no_poster_does_not_raise():
    assert post_failure("doing something", RuntimeError("boom"), None, source="S260814a") is None


def test_post_failure_writes_an_offline_payload(config):
    poster = SlackPoster(config)
    posted = post_failure("handling an alert notice", RuntimeError("boom"), poster, source="S260814a")
    assert posted is not None
    assert posted.offline


def config_work_posts_dir(config):
    from pathlib import Path

    return Path(config.paths.work_dir).expanduser() / "posts"


class _FakeHeartbeat:
    """Stands in for `decam_chatterbox.heartbeat.Heartbeat`.

    The heartbeat's own timer/threading behaviour is covered in
    test_heartbeat.py; these tests only need to know `run_service` starts and
    stops one.
    """

    calls: list[str] = []

    def __init__(self, config, poster, status):
        type(self).calls.append("init")

    def start(self):
        type(self).calls.append("start")

    def stop(self, timeout=5.0):
        type(self).calls.append("stop")


def test_run_service_replay_handles_every_record_and_returns_the_count(tmp_path, notice, config):
    path1 = tmp_path / "a.json"
    path2 = tmp_path / "b.json"
    path1.write_text(json.dumps(notice))
    path2.write_text(json.dumps(make_notice(superevent_id="S260814b")))

    handled = run_service(config, paths=[path1, path2])
    assert handled == 2


def test_run_service_starts_and_stops_the_heartbeat(tmp_path, notice, config, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))
    _FakeHeartbeat.calls = []
    monkeypatch.setattr(app_module, "Heartbeat", _FakeHeartbeat)

    run_service(config, paths=[path])
    assert _FakeHeartbeat.calls == ["init", "start", "stop"]


def test_run_service_posts_shutdown_on_keyboard_interrupt(tmp_path, notice, config, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))

    def raise_interrupt(record, config, poster=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr(app_module, "process_notice", raise_interrupt)
    handled = run_service(config, paths=[path])
    assert handled == 0

    payload = json.loads((config_work_posts_dir(config) / "shutdown.json").read_text())
    assert "shutting down" in payload["text"].lower()
    assert "interrupted" in payload["text"].lower()


def test_run_service_installs_a_sigterm_handler_like_ctrl_c(tmp_path, notice, config, monkeypatch):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))
    calls = []
    monkeypatch.setattr(signal_module, "signal", lambda *args: calls.append(args))

    run_service(config, paths=[path])
    assert calls
    assert calls[0] == (signal_module.SIGTERM, signal_module.default_int_handler)


def test_run_service_tolerates_signal_registration_failure(tmp_path, notice, config, monkeypatch):
    """Registering SIGTERM only works on the main thread; elsewhere it must
    degrade quietly rather than take the whole service down."""

    def raise_value_error(*args):
        raise ValueError("signal only works in main thread of the main interpreter")

    monkeypatch.setattr(signal_module, "signal", raise_value_error)
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))
    assert run_service(config, paths=[path]) == 1
