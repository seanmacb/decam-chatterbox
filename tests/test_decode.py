"""Tests for decam_chatterbox.ingest.decode."""

import json

import pytest
from conftest import make_notice

from decam_chatterbox.ingest.decode import decode_notice, load_record_file
from decam_chatterbox.models import Trigger


def test_decode_preliminary_notice(notice):
    trigger = decode_notice(notice)
    assert isinstance(trigger, Trigger)
    assert trigger.superevent_id == "S260814a"
    assert trigger.alert_type == "PRELIMINARY"
    assert trigger.is_real
    assert not trigger.is_retraction
    assert not trigger.is_burst
    assert trigger.group == "CBC"
    assert trigger.instruments == ["H1", "L1"]
    assert trigger.classification["BNS"] == pytest.approx(0.9)
    assert trigger.most_likely_class == ("BNS", pytest.approx(0.9))
    assert trigger.far_hz == pytest.approx(1e-10)
    assert trigger.inverse_far_years is not None and trigger.inverse_far_years > 0


def test_decode_populates_localization(notice):
    trigger = decode_notice(notice)
    assert trigger.localization is not None
    assert trigger.geometry is not None
    assert trigger.localization_error is None
    assert trigger.geometry.area_deg2 > 0
    assert trigger.distance_mean_mpc == pytest.approx(120.0)
    assert trigger.distance_std_mpc == pytest.approx(30.0)
    # The disc was centred at ra=60, dec=-40 in conftest.make_skymap_bytes.
    assert trigger.geometry.centroid_ra_deg == pytest.approx(60.0, abs=2.0)
    assert trigger.geometry.centroid_dec_deg == pytest.approx(-40.0, abs=2.0)


def test_decode_avro_style_raw_bytes_skymap(notice_avro, notice):
    """Avro (SCIMMA) notices carry raw bytes; JSON (GCN) notices base64 text.

    Both must decode to the same localization.
    """
    trigger_avro = decode_notice(notice_avro)
    trigger_json = decode_notice(notice)
    assert trigger_avro.geometry.area_deg2 == pytest.approx(trigger_json.geometry.area_deg2)
    assert trigger_avro.distance_mean_mpc == pytest.approx(trigger_json.distance_mean_mpc)


def test_decode_retraction_has_no_event_fields(retraction_notice):
    trigger = decode_notice(retraction_notice)
    assert trigger.is_retraction
    assert trigger.localization is None
    assert trigger.geometry is None
    assert trigger.classification == {}
    assert trigger.far_hz is None


def test_decode_mock_event_is_not_real(mock_notice):
    trigger = decode_notice(mock_notice)
    assert not trigger.is_real
    assert trigger.search == "MDC"


def test_decode_burst_event_has_no_classification():
    notice = make_notice(
        group="Burst",
        pipeline="cWB",
        search="AllSky",
        classification={},
        properties={},
    )
    notice["event"]["duration"] = 0.2
    notice["event"]["central_frequency"] = 150.0
    trigger = decode_notice(notice)
    assert trigger.is_burst
    assert trigger.classification == {}
    assert trigger.most_likely_class is None
    assert trigger.duration_s == pytest.approx(0.2)
    assert trigger.central_frequency_hz == pytest.approx(150.0)


def test_decode_missing_required_field_raises():
    with pytest.raises(KeyError):
        decode_notice({"alert_type": "PRELIMINARY", "superevent_id": "S260814a"})


def test_decode_null_superevent_id_raises():
    with pytest.raises(ValueError):
        decode_notice({"alert_type": "PRELIMINARY", "superevent_id": None, "time_created": "x"})


def test_decode_missing_skymap_leaves_localization_none():
    notice = make_notice(skymap_bytes=None)
    trigger = decode_notice(notice)
    assert trigger.localization is None
    assert trigger.localization_error is None  # not an error: this notice never had one


def test_decode_corrupt_skymap_is_non_fatal():
    notice = make_notice(skymap_bytes=b"not a fits file", skymap_encoding="raw")
    trigger = decode_notice(notice)
    assert trigger.localization is None
    assert trigger.localization_error is not None


def test_decode_saves_skymap_to_disk_when_given_a_directory(notice, tmp_path):
    decode_notice(notice, skymap_dir=tmp_path)
    expected = tmp_path / "S260814a_preliminary.fits"
    assert expected.is_file()


def test_decode_external_coincidence():
    notice = make_notice(
        external_coinc={
            "gcn_notice_id": 12345,
            "ivorn": "ivo://nasa.gsfc.gcn/Fermi#GBM_Fin_Pos",
            "observatory": "Fermi",
            "search": "GRB",
            "time_difference": 2.02,
            "time_coincidence_far": 1e-14,
            "time_sky_position_coincidence_far": 1e-16,
            "combined_skymap": "not decoded",
        }
    )
    trigger = decode_notice(notice)
    assert trigger.external_coinc is not None
    assert trigger.external_coinc.observatory == "Fermi"
    assert trigger.external_coinc.has_combined_skymap is True
    # The combined skymap payload itself must not leak into `raw`.
    assert "combined_skymap" not in trigger.raw["external_coinc"]


def test_load_record_file_json(tmp_path, notice):
    path = tmp_path / "record.json"
    path.write_text(json.dumps(notice))
    loaded = load_record_file(path)
    assert loaded["superevent_id"] == notice["superevent_id"]


def test_load_record_file_json_list_envelope(tmp_path, notice):
    path = tmp_path / "record.json"
    path.write_text(json.dumps([notice]))
    loaded = load_record_file(path)
    assert loaded["superevent_id"] == notice["superevent_id"]


def test_load_record_file_unsupported_suffix(tmp_path):
    path = tmp_path / "record.xml"
    path.write_text("<x/>")
    with pytest.raises(ValueError):
        load_record_file(path)
