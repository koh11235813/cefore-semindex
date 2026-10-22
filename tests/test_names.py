"""Tests for the Sync protocol wire names (state and record)."""

from __future__ import annotations

import pytest

from semindex import names


def test_registration_prefix():
    assert names.registration_prefix("g", "h0") == "ccnx:/semidx/g/h0"


def test_state_name_round_trip():
    built = names.build_state_name("g", "h1", "h0", 1700000000000)
    assert built == "ccnx:/semidx/g/h1/sv/h0/1700000000000"
    parsed = names.parse(built)
    assert parsed == names.StateName(group="g", holder="h1", requester="h0", epoch_ms=1700000000000)


def test_record_name_round_trip():
    built = names.build_record_name("g", "h1", "h0", 3)
    assert built == "ccnx:/semidx/g/h1/rec/h0/3"
    parsed = names.parse(built)
    assert parsed == names.RecordName(group="g", holder="h1", origin="h0", seq=3)


def test_build_record_name_rejects_seq_below_one():
    with pytest.raises(names.MalformedNameError) as exc_info:
        names.build_record_name("g", "h1", "h0", 0)
    assert str(exc_info.value) == "seq must be >= 1"


def test_build_state_name_rejects_negative_epoch_ms():
    with pytest.raises(names.MalformedNameError) as exc_info:
        names.build_state_name("g", "h1", "h0", -1)
    assert str(exc_info.value) == "epoch_ms must be non-negative"


def test_build_rejects_component_containing_slash():
    with pytest.raises(names.MalformedNameError) as exc_info:
        names.build_record_name("g", "h1/evil", "h0", 1)
    assert str(exc_info.value) == "holder must be a non-empty name component without '/'"


@pytest.mark.parametrize(
    "name,expected_message",
    [
        ("ccnx:/other/g/h1/sv/h0/1", "name must start with 'ccnx:/semidx/'"),
        ("ccnx:/semidx/g/h1/sv/h0", "name must have exactly 5 segments after the fixed prefix"),
        ("ccnx:/semidx/g/h1/sv/h0/1/extra", "name must have exactly 5 segments after the fixed prefix"),
        ("ccnx:/semidx/g/h1/wrong/h0/1", "name kind segment must be 'sv' or 'rec'"),
        ("ccnx:/semidx/g/h1/sv/h0/abc", "epoch_ms must be a non-negative decimal integer"),
        ("ccnx:/semidx/g/h1/rec/h0/abc", "seq must be a non-negative decimal integer"),
        ("ccnx:/semidx/g/h1/rec/h0/0", "seq must be >= 1"),
        ("ccnx:/semidx/g//sv/h0/1", "holder must be a non-empty name component without '/'"),
        ("ccnx:/semidx/g/h1/sv/h0/-1", "epoch_ms must be a non-negative decimal integer"),
    ],
)
def test_parse_rejects_malformed_names(name, expected_message):
    with pytest.raises(names.MalformedNameError) as exc_info:
        names.parse(name)
    assert str(exc_info.value) == expected_message
