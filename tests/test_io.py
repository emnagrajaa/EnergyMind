"""Tests for Step 1 (io.py): derived-column drop, UTC conversion, labels split."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from energymind import io
from energymind.config import REPO_ROOT, load_settings

FIXTURE = Path(__file__).parent / "fixtures" / "sample_two_households.csv"
RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"

DERIVED_LEAKS = [
    "power_2h_mean",
    "power_7d_std",
    "power_lag_1",
    "power_zscore",
    "is_demand_spike",
    "is_statistical_anomaly",
    "cumulative_daily_kwh",
]


@pytest.fixture(scope="module")
def cfg():
    return load_settings().data


@pytest.fixture(scope="module")
def fixture_frame() -> pd.DataFrame:
    return io.load_raw(FIXTURE)


def test_derived_block_start_and_size(cfg, fixture_frame):
    columns = list(fixture_frame.columns)
    assert len(columns) == 104
    # `hour` is the first derived column and the boundary of the measured block.
    assert columns.index(cfg.columns.derived_start) == 40
    # 64 positional columns from `hour` onward; tariff_rate is an input kept in
    # the measured block, so 63 are actually dropped.
    positional = len(columns) - columns.index(cfg.columns.derived_start)
    assert positional == 64
    assert len(io.derived_columns(columns, cfg)) == 63
    assert "tariff_rate" not in io.derived_columns(columns, cfg)


def test_measured_excludes_every_derived_column(cfg, fixture_frame):
    columns = list(fixture_frame.columns)
    measured = io.select_measured_columns(columns, cfg)
    assert set(measured).isdisjoint(io.derived_columns(columns, cfg))
    for leak in DERIVED_LEAKS:
        assert leak not in measured


def test_measured_drops_labels_and_adds_back(cfg, fixture_frame):
    measured = io.select_measured_columns(list(fixture_frame.columns), cfg)
    assert "anomaly_flag" not in measured
    assert "anomaly_type" not in measured
    assert "renewable_contribution_pct" not in measured
    assert "tariff_rate" in measured  # an input, re-appended
    assert len(measured) == 38  # 37 measured/metadata + tariff_rate


def test_utc_conversion_is_offset_aware(cfg, fixture_frame):
    df = io.add_utc(fixture_frame, cfg)
    assert str(df["ts_utc"].dt.tz) == "UTC"
    # Local 2020-01-01 00:00 with offset -6 -> 06:00 UTC.
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2020-01-01 06:00:00", tz="UTC")


def test_provided_labels_are_split_out(cfg, fixture_frame):
    df = io.add_utc(fixture_frame, cfg)
    labels = io.split_provided_labels(df, cfg)
    assert list(labels.columns) == ["household_id", "ts_utc", "anomaly_type"]
    assert len(labels) == 2
    assert set(labels["anomaly_type"]) == {"spike", "dropout"}


def test_load_measured_returns_clean_frame(cfg):
    measured, labels = io.load_measured(FIXTURE, cfg)
    assert len(measured) == 12
    assert len(labels) == 2
    assert "ts_utc" in measured.columns
    assert measured["ts_utc"].dt.tz is not None


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n", encoding="utf-8")
    manifest = tmp_path / "MANIFEST.csv"
    io.write_manifest(csv, manifest, rows=1, columns=2)
    assert io.verify_manifest(csv, manifest)["verified"] is True

    csv.write_text("a,b\n1,3\n", encoding="utf-8")  # tamper
    assert io.verify_manifest(csv, manifest)["verified"] is False


def test_missing_raw_csv_raises(cfg, tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_raw(tmp_path / "nope.csv")


# --------------------------------------------------------------------------- #
# Real dataset (skipped in CI, where the 184 MB CSV is absent)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_dataset_step1(cfg):
    measured, labels = io.load_measured(RAW_CSV, cfg)
    assert len(measured) == 227_752
    assert measured["household_id"].nunique() == 13
    assert len(labels) == 698
    assert set(labels["anomaly_type"]) == {"spike", "dropout"}
    counts = labels["anomaly_type"].value_counts()
    assert counts["spike"] == 490
    assert counts["dropout"] == 208


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_dataset_manifest_matches():
    settings = load_settings()
    result = io.verify_manifest(settings.data.paths.raw_csv, settings.data.paths.manifest)
    assert result["verified"] is True
