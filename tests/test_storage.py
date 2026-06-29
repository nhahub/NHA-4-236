"""Tests for the JSON user-profile store."""
from __future__ import annotations

import pytest

import storage
from patient import PatientInfo


@pytest.fixture(autouse=True)
def _tmp_users_dir(tmp_path, monkeypatch):
    # Redirect the store to a temp dir so tests don't touch real data/users.
    monkeypatch.setattr(storage, "USERS_DIR", tmp_path)
    yield


def test_save_and_load_round_trip():
    p = PatientInfo(age=65, sex="female", conditions="diabetes", pregnancy="no")
    storage.save_profile("Alex R", p)
    loaded = storage.load_profile("Alex R")
    assert loaded == p


def test_load_missing_returns_none():
    assert storage.load_profile("nobody") is None


def test_ids_are_filename_sanitized():
    storage.save_profile("Alex R!!", PatientInfo(age=30))
    # "Alex R!!" -> "alex-r"; same slug loads it back.
    assert storage.load_profile("alex r").age == 30
    assert "alex-r" in storage.list_profiles()


def test_load_ignores_unknown_fields(tmp_path):
    # A file with an extra/old field must still load (forward/backward safe).
    import json

    (tmp_path / "x.json").write_text(
        json.dumps({"patient": {"age": 40, "legacy_field": "x"}}), encoding="utf-8"
    )
    assert storage.load_profile("x").age == 40
