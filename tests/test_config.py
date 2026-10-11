"""Tests for typed config loading and validation (M0)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from energymind.config import (
    DEFAULT_CONFIGS_DIR,
    REPO_ROOT,
    DataConfig,
    load_settings,
    load_yaml,
)


@pytest.fixture(scope="module")
def settings():
    return load_settings()


def test_default_config_dir_exists():
    assert DEFAULT_CONFIGS_DIR.is_dir()
    assert DEFAULT_CONFIGS_DIR == REPO_ROOT / "configs"


def test_all_config_files_present():
    for name in ("data", "injection", "models", "policy", "agents", "store"):
        assert (DEFAULT_CONFIGS_DIR / f"{name}.yaml").is_file()


def test_settings_load(settings):
    assert settings.data.paths.raw_csv.endswith("Austin_TX_US_2020.csv")
    assert settings.data.time.expected_rows_per_household == 17_520
    assert settings.data.time.assume_utc_offset_h == -6


def test_injection_types_and_origins(settings):
    typed = settings.injection.types
    assert len(typed) == 15
    origins = {t.origin for t in typed.values()}
    assert origins == {"data_quality", "adversarial", "physical_event"}
    # Replay and mimicry must be withheld so the "unknown" path can be tested.
    assert {"replay", "mimicry"} <= set(settings.injection.withhold_from_train)
    assert all(p in (0, 1) or 0 < p < 1 for p in settings.injection.prevalence)


def test_splits_are_ordered_and_contiguous(settings):
    tr = settings.data.splits.temporal
    assert tr.train.start < tr.train.end <= tr.val.start
    assert tr.val.end <= tr.test.start


def test_policy_thresholds_ordered(settings):
    tiers = settings.policy.tiers
    assert tiers.tau_auto >= tiers.tau_review
    assert all(spec.risk in {"low", "medium", "high"} for spec in settings.policy.actions.values())


def test_agents_use_distinct_families(settings):
    assert settings.agents.analyst.family != settings.agents.validator.family
    assert settings.agents.analyst.model != settings.agents.validator.model


def test_enabled_detectors_include_required_baselines(settings):
    enabled = {n for n, s in settings.models.detectors.items() if s.enabled}
    assert {"robust_z", "rules", "iforest"} <= enabled
    assert settings.models.evaluation.no_point_adjust is True


def test_unknown_key_is_rejected(tmp_path):
    bad = tmp_path / "data.yaml"
    bad.write_text(
        "paths: {raw_csv: x, manifest: y, clean_dir: a, features_dir: b, "
        "injected_dir: c, splits_file: d}\n"
        "time: {}\ncolumns: {}\n"
        "bounds: {voltage_v: [207, 253], frequency_hz: [49.5, 50.5], "
        "power_factor: [0, 1], active_power_kw: [0, null], sub_meter_tolerance: 0.05}\n"
        "physics_residual_tolerance: 0.01\n"
        "splits: {temporal: {train: {start: 2020-01-01, end: 2020-09-01}, "
        "val: {start: 2020-09-01, end: 2020-11-01}, "
        "test: {start: 2020-11-01, end: 2020-12-31}}}\n"
        "typo_key: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        DataConfig.model_validate(load_yaml(bad))


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path)
