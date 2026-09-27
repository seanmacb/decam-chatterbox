"""Tests for decam_chatterbox.ingest.source."""

import json

import pytest

from decam_chatterbox.config import Config
from decam_chatterbox.ingest.source import FileAlertSource, ReplaySource, make_source


def test_replay_source_yields_each_path_in_order(tmp_path, notice, retraction_notice):
    path1 = tmp_path / "a.json"
    path2 = tmp_path / "b.json"
    path1.write_text(json.dumps(notice))
    path2.write_text(json.dumps(retraction_notice))

    source = ReplaySource([path1, path2])
    records = list(source)
    superevent_ids = [r["superevent_id"] for r, _ in records]
    assert superevent_ids == [notice["superevent_id"], retraction_notice["superevent_id"]]
    assert records[0][1]["transport"] == "replay"


def test_file_source_picks_up_new_files_once(tmp_path, notice):
    (tmp_path / "s1.json").write_text(json.dumps(notice))
    source = FileAlertSource(str(tmp_path), once=True)
    records = list(source)
    assert len(records) == 1
    assert records[0][1]["transport"] == "files"

    # A second pass with `once=True` and no new files yields nothing.
    source2 = FileAlertSource(str(tmp_path), once=True)
    # Simulate having already seen it by re-scanning: a fresh source has not,
    # so it is picked up again -- only the *same* source instance dedupes.
    assert len(list(source2)) == 1


def test_file_source_does_not_redeliver_within_one_instance(tmp_path, notice):
    (tmp_path / "s1.json").write_text(json.dumps(notice))
    source = FileAlertSource(str(tmp_path), once=True)
    list(source)
    assert list(source) == []


def test_make_source_files_kind(tmp_path):
    config = Config()
    config.ingest.kind = "files"
    config.ingest.watch_dir = str(tmp_path)
    source = make_source(config)
    assert isinstance(source, FileAlertSource)


def test_make_source_replay_kind_without_paths_raises():
    config = Config()
    config.ingest.kind = "replay"
    with pytest.raises(ValueError):
        make_source(config)


def test_make_source_unknown_kind_raises():
    config = Config()
    config.ingest.kind = "carrier-pigeon"
    with pytest.raises(ValueError):
        make_source(config)


def test_make_source_explicit_paths_force_replay(tmp_path, notice):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(notice))
    config = Config()
    config.ingest.kind = "scimma"  # ignored: explicit paths win
    source = make_source(config, paths=[path])
    assert isinstance(source, ReplaySource)


def test_scimma_source_requires_a_url():
    from decam_chatterbox.ingest.source import ScimmaAlertSource

    with pytest.raises(ValueError):
        ScimmaAlertSource(url="")
