"""End-to-end tests for decam_chatterbox.app, offline throughout."""

from conftest import make_notice

from decam_chatterbox.app import TriggerReport, post_failure, process_notice
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
