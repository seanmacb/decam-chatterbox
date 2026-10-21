"""Tests for tagging the shift crew on a high-priority alert."""

import json

import pytest
from astropy.time import TimeDelta
from conftest import make_notice

from decam_chatterbox.app import process_notice
from decam_chatterbox.config import load_config
from decam_chatterbox.doctor import _slack_crew_check
from decam_chatterbox.ingest.decode import decode_notice
from decam_chatterbox.priority import assess_priority
from decam_chatterbox.slackbot.blocks import build_trigger_blocks, crew_applies, crew_blocks
from decam_chatterbox.slackbot.client import SlackPoster, render_blocks_as_text

CREW_IDS = {"decam-too-lead": "S0LEAD", "decam-observers": "S0OBS", "decam-support": "S0SUP"}


@pytest.fixture
def crew_config(config):
    config.slack.crew.lead = "decam-too-lead"
    config.slack.crew.notify = ["decam-observers", "decam-support"]
    return config


def fresh_trigger(notice):
    """A trigger received a minute after its event, so it is not stale."""
    event = decode_notice(notice).event_time
    return decode_notice(notice, received_at=event + TimeDelta(60.0, format="sec"))


def crew_text(trigger, config, ids=CREW_IDS, priority=None) -> str:
    if priority is None:
        priority = assess_priority(trigger, config.priority)
    return render_blocks_as_text(crew_blocks(trigger, config, ids, priority))


def test_the_default_notice_is_high_priority(notice, crew_config):
    """Everything below relies on the fixture clearing every threshold."""
    assert assess_priority(fresh_trigger(notice), crew_config.priority).is_high_priority


def test_a_high_priority_alert_tags_every_group(notice, crew_config):
    text = crew_text(fresh_trigger(notice), crew_config)
    for group_id in CREW_IDS.values():
        assert f"<!subteam^{group_id}>" in text


def test_the_lead_is_told_they_decide(notice, crew_config):
    text = crew_text(fresh_trigger(notice), crew_config)
    lead_line = next(line for line in text.splitlines() if "S0LEAD" in line)
    assert "ToO lead" in lead_line
    assert "authority to make decisions" in lead_line
    assert "S0OBS" not in lead_line and "S0SUP" not in lead_line


def test_the_lead_role_is_configurable(notice, crew_config):
    crew_config.slack.crew.lead_role = "DECam ToO coordinator"
    text = crew_text(fresh_trigger(notice), crew_config)
    assert "you are the DECam ToO coordinator" in text
    assert "contact with your DECam ToO coordinator" in text


def test_the_others_are_told_to_wait_for_the_lead(notice, crew_config):
    text = crew_text(fresh_trigger(notice), crew_config)
    notify_line = next(line for line in text.splitlines() if "S0OBS" in line)
    assert "S0SUP" in notify_line, "addressed together"
    assert "be aware" in notify_line
    assert "as soon as possible" in notify_line
    assert "await further instructions" in notify_line
    assert "S0LEAD" not in notify_line


def test_every_group_is_tagged_once(notice, crew_config):
    crew_config.slack.crew.notify = ["decam-observers", "decam-too-lead", "decam-support"]
    text = crew_text(fresh_trigger(notice), crew_config)
    assert text.count("S0LEAD") == 1, "the lead is not also told to wait for themselves"


def test_nobody_is_tagged_until_a_crew_is_configured(notice, config):
    """No invented handles: the default is inert."""
    trigger = fresh_trigger(notice)
    assert config.slack.crew.handles() == []
    assert crew_blocks(trigger, config, CREW_IDS, assess_priority(trigger, config.priority)) == []


def test_a_routine_alert_carries_no_crew_line_at_all(crew_config):
    """Not high priority: the same rule as slack.mention, and no clutter."""
    from conftest import make_skymap_bytes

    notice = make_notice(skymap_bytes=make_skymap_bytes(), far=1e-5)
    trigger = fresh_trigger(notice)
    priority = assess_priority(trigger, crew_config.priority)
    assert not priority.is_high_priority
    assert crew_blocks(trigger, crew_config, CREW_IDS, priority) == []
    assert not crew_applies(trigger, crew_config, priority)


def test_every_real_alert_can_be_tagged_instead(crew_config):
    from conftest import make_skymap_bytes

    crew_config.slack.crew.only_high_priority = False
    notice = make_notice(skymap_bytes=make_skymap_bytes(), far=1e-5)
    assert "<!subteam^S0LEAD>" in crew_text(fresh_trigger(notice), crew_config)


def test_without_a_priority_assessment_nobody_is_tagged(notice, crew_config):
    trigger = fresh_trigger(notice)
    assert crew_blocks(trigger, crew_config, CREW_IDS, None) == []


def test_a_mock_alert_tags_nobody_and_says_so(mock_notice, crew_config):
    text = crew_text(fresh_trigger(mock_notice), crew_config)
    assert "<!subteam" not in text
    assert "shift was not tagged" in text
    assert "@decam-too-lead" in text, "plain text, which pings nobody"


def test_a_retraction_tags_nobody_and_says_nothing(retraction_notice, crew_config):
    trigger = decode_notice(retraction_notice)
    assert crew_blocks(trigger, crew_config, CREW_IDS, assess_priority(trigger, crew_config.priority)) == []


def test_an_old_event_tags_nobody_and_says_so(notice, crew_config):
    """A replay of an archived alert must not call the shift to action."""
    stale = decode_notice(notice)  # received now; the fixture event is from the past
    assert stale.age_s > 24 * 3600
    text = crew_text(stale, crew_config)
    assert "<!subteam" not in text
    assert "old, so the shift was not tagged" in text


def test_the_age_limit_can_be_switched_off(notice, crew_config):
    crew_config.slack.crew.max_age_hours = 0
    assert "<!subteam^S0LEAD>" in crew_text(decode_notice(notice), crew_config)


def test_an_event_of_unknown_age_is_still_tagged(notice, crew_config):
    """When the age cannot be told, failing to tag is the worse error."""
    trigger = fresh_trigger(notice)
    priority = assess_priority(trigger, crew_config.priority)
    trigger.event_time = None
    assert "<!subteam^S0LEAD>" in crew_text(trigger, crew_config, priority=priority)


def test_a_group_without_an_id_is_written_as_text_and_flagged(notice, crew_config):
    """A handle typed as text pings nobody; the post must not look tagged."""
    text = crew_text(fresh_trigger(notice), crew_config, ids={"decam-too-lead": "S0LEAD"})
    assert "<!subteam^S0LEAD>" in text
    assert "@decam-observers" in text and "<!subteam^S0OBS>" not in text
    assert "Not notified" in text
    assert "@decam-observers, @decam-support" in text.split("Not notified")[1]


def test_a_dry_run_does_not_claim_the_post_would_not_notify(notice, crew_config):
    """ids=None means 'not looked up', which is not 'not found'."""
    trigger = fresh_trigger(notice)
    text = render_blocks_as_text(
        crew_blocks(trigger, crew_config, None, assess_priority(trigger, crew_config.priority))
    )
    assert "@decam-too-lead" in text
    assert "Not notified" not in text


def test_the_crew_can_be_switched_off(notice, crew_config):
    crew_config.slack.crew.enabled = False
    trigger = fresh_trigger(notice)
    assert crew_blocks(trigger, crew_config, CREW_IDS, assess_priority(trigger, crew_config.priority)) == []


def test_the_crew_comes_straight_after_the_header(notice, crew_config):
    """It is the part of the post somebody has to act on."""
    trigger = fresh_trigger(notice)
    priority = assess_priority(trigger, crew_config.priority)
    blocks = build_trigger_blocks(
        trigger, None, None, None, crew_config, priority=priority, crew_ids=CREW_IDS
    )
    # Header, the event summary, the priority "why" line, then the crew --
    # ahead of the divider and everything else.
    assert blocks[0]["type"] == "header"
    assert trigger.superevent_id in blocks[1]["text"]["text"]
    assert blocks[2]["type"] == "context" and "Priority triage" in blocks[2]["elements"][0]["text"]
    assert "S0LEAD" in blocks[3]["text"]["text"]


# --------------------------------------------------------------------- poster


def test_ids_from_the_config_are_used_without_asking_slack(crew_config, tmp_path):
    crew_config.slack.crew.ids = dict(CREW_IDS)
    poster = SlackPoster(crew_config, dry_run=True, output_dir=tmp_path)
    assert poster.crew_ids() == CREW_IDS


def test_handles_are_resolved_through_usergroups_list(crew_config, tmp_path, monkeypatch):
    calls = []

    class FakeClient:
        def usergroups_list(self):
            calls.append(1)
            return {
                "usergroups": [
                    {"handle": "decam-too-lead", "id": "S0LEAD"},
                    {"handle": "decam-observers", "id": "S0OBS"},
                    {"handle": "decam-support", "id": "S0SUP"},
                    {"handle": "unrelated", "id": "S0OTHER"},
                ]
            }

    monkeypatch.setenv(crew_config.slack.bot_token_env, "xoxb-test")
    poster = SlackPoster(crew_config, output_dir=tmp_path)
    monkeypatch.setattr(poster, "_get_client", lambda: FakeClient())

    assert poster.crew_ids() == CREW_IDS
    assert poster.crew_ids() == CREW_IDS
    assert len(calls) == 1, "looked up once, not once per alert"


def test_a_failed_lookup_loses_the_tag_and_not_the_alert(crew_config, tmp_path, monkeypatch, caplog):
    class Refusing:
        def usergroups_list(self):
            raise RuntimeError("missing_scope")

    monkeypatch.setenv(crew_config.slack.bot_token_env, "xoxb-test")
    crew_config.slack.crew.ids = {"decam-too-lead": "S0LEAD"}
    poster = SlackPoster(crew_config, output_dir=tmp_path)
    monkeypatch.setattr(poster, "_get_client", lambda: Refusing())

    with caplog.at_level("ERROR"):
        ids = poster.crew_ids()
    assert ids == {"decam-too-lead": "S0LEAD"}, "what is known survives"
    assert "usergroups:read" in caplog.text


def test_an_offline_poster_cannot_look_anything_up(crew_config, tmp_path):
    assert SlackPoster(crew_config, dry_run=True, output_dir=tmp_path).crew_ids() is None


def test_the_tag_reaches_the_posted_payload(notice, crew_config, tmp_path):
    """Through the poster: what goes to Slack carries the mention."""
    trigger = fresh_trigger(notice)
    blocks = crew_blocks(trigger, crew_config, CREW_IDS, assess_priority(trigger, crew_config.priority))
    poster = SlackPoster(crew_config, dry_run=True, output_dir=tmp_path)
    poster.post(blocks, "alert", label="crew")
    payload = json.loads((tmp_path / "crew.json").read_text())
    assert "<!subteam^S0LEAD>" in payload["blocks"][0]["text"]["text"]


def test_process_notice_tags_from_configured_ids(notice, crew_config, tmp_path):
    """End to end: ids from the config reach the post even offline."""
    crew_config.slack.crew.ids = dict(CREW_IDS)
    crew_config.slack.crew.max_age_hours = 0  # the fixture event is months old
    poster = SlackPoster(crew_config, dry_run=True, output_dir=tmp_path / "posts")
    report = process_notice(notice, crew_config, poster=poster, out_dir=tmp_path / "plots")
    assert "<!subteam^S0LEAD>" in render_blocks_as_text(report.blocks)


# ---------------------------------------------------------- config and doctor


def test_the_crew_loads_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "slack:\n"
        "  crew:\n"
        "    lead: decam-too-lead\n"
        "    lead_role: ToO coordinator\n"
        "    notify: [decam-observers]\n"
        "    ids: {decam-too-lead: S0LEAD}\n"
        "    only_high_priority: false\n"
    )
    crew = load_config(path).slack.crew
    assert crew.handles() == ["decam-too-lead", "decam-observers"]
    assert crew.lead_role == "ToO coordinator"
    assert crew.ids == {"decam-too-lead": "S0LEAD"}
    assert crew.only_high_priority is False


def test_doctor_is_happy_with_no_crew_configured(config):
    check = _slack_crew_check(config)
    assert check.ok
    assert "nobody configured" in check.detail


def test_doctor_flags_a_crew_it_cannot_resolve(crew_config):
    check = _slack_crew_check(crew_config)
    assert not check.ok
    assert "without a Slack token" in check.detail


def test_doctor_is_happy_when_every_id_is_given(crew_config):
    crew_config.slack.crew.ids = dict(CREW_IDS)
    assert _slack_crew_check(crew_config).ok
