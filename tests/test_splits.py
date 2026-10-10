"""Tests for Step 4 (splits.py): temporal windows, hold-out, experience ids."""

from __future__ import annotations

import pandas as pd
import pytest

from energymind import clean, splits
from energymind.config import REPO_ROOT, load_settings

RAW_CSV = REPO_ROOT / "data" / "raw" / "Austin_TX_US_2020.csv"


def _cfg():
    return load_settings().data


def _ts(*specs: str) -> pd.DataFrame:
    return pd.DataFrame({"ts_utc": pd.to_datetime(list(specs), utc=True)})


# --------------------------------------------------------------------------- #
# Temporal assignment
# --------------------------------------------------------------------------- #


def test_temporal_windows_are_ordered_and_contiguous():
    bounds = splits.temporal_bounds(_cfg())
    assert bounds["train"][1] == bounds["val"][0]
    assert bounds["val"][1] == bounds["test"][0]
    assert bounds["train"][0] < bounds["train"][1]


def test_assign_temporal_labels_and_boundaries():
    cfg = _cfg()
    df = _ts(
        "2020-01-01 06:00",  # local Jan 1 00:00 -> train
        "2020-06-15 12:00",  # train
        "2020-09-01 06:00",  # local Sep 1 00:00 -> val (boundary)
        "2020-10-15 00:00",  # val
        "2020-11-01 06:00",  # local Nov 1 00:00 -> test (boundary)
        "2020-12-30 05:30",  # test (last grid slot)
    )
    assert splits.assign_temporal(df, cfg).tolist() == [
        "train",
        "train",
        "val",
        "val",
        "test",
        "test",
    ]


def test_assign_temporal_leaves_out_of_range_unassigned():
    cfg = _cfg()
    df = _ts("2021-01-01 00:00", "2019-12-31 00:00")
    assert splits.assign_temporal(df, cfg).isna().all()


def test_experience_id_is_local_month():
    cfg = _cfg()
    # 2020-01-01 02:00 UTC = 2020-12-31 20:00 local -> month 12
    df = _ts("2020-01-01 06:00", "2020-01-01 02:00", "2020-07-15 06:00")
    assert splits.experience_ids(df, cfg).tolist() == [1, 12, 7]


# --------------------------------------------------------------------------- #
# Hold-out selection
# --------------------------------------------------------------------------- #


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "household_id": ["A", "B", "C", "D", "E"],
            "household_type": [
                "commercial_small",
                "household",
                "household",
                "household",
                "household",
            ],
            "has_solar": [0, 1, 0, 0, 0],
            "has_ev": [0, 0, 1, 0, 0],
            "has_battery": [0, 0, 0, 0, 0],
        }
    )


def test_choose_holdout_covers_one_of_each_category():
    chosen = splits.choose_holdout(_meta(), _cfg())
    assert chosen == ["A", "B", "C"]  # commercial, solar, EV; sorted


def test_choose_holdout_respects_explicit_config():
    cfg = _cfg()
    configured = cfg.model_copy(
        update={"splits": cfg.splits.model_copy(update={"holdout_households": ["D", "E"]})}
    )
    assert splits.choose_holdout(_meta(), configured) == ["D", "E"]


def test_household_roles_partition():
    roles = splits.household_roles(["A", "B", "C"], holdout=["B"])
    assert roles.to_dict() == {"A": "train_val", "B": "test", "C": "train_val"}


def test_add_split_columns_partitions_households():
    cfg = _cfg()
    df = pd.DataFrame(
        {
            "household_id": ["H1", "H1", "H2", "H2"],
            "ts_utc": pd.to_datetime(
                ["2020-03-01 12:00", "2020-11-15 12:00", "2020-03-01 12:00", "2020-11-15 12:00"],
                utc=True,
            ),
        }
    )
    payload = {"holdout_households": ["H2"], "train_val_households": ["H1"]}
    out = splits.add_split_columns(df, payload, cfg)
    assert out["split"].tolist() == ["train", "test", "train", "test"]
    assert out["household_role"].tolist() == ["train_val", "train_val", "test", "test"]
    # a household never appears in both roles
    assert out.groupby("household_id")["household_role"].nunique().eq(1).all()


# --------------------------------------------------------------------------- #
# Real dataset (skipped in CI)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.skipif(not RAW_CSV.exists(), reason="raw dataset not present")
def test_real_splits_are_complete_and_disjoint():
    settings = load_settings()
    payload = splits.build_splits(settings)

    assert len(payload["holdout_households"]) == 3
    assert len(payload["train_val_households"]) == 10
    assert not set(payload["holdout_households"]) & set(payload["train_val_households"])

    meta = splits.load_households(settings).set_index("household_id")
    hold = meta.loc[payload["holdout_households"]]
    assert (hold["has_solar"] == 1).any()
    assert (hold["has_ev"] == 1).any()
    assert (hold["household_type"] == "commercial_small").any()

    df = clean.load_baseline(settings)
    report = splits.verify(df, settings.data)
    assert report["temporal_unassigned"] == 0
    assert sum(report["per_split"].values()) == report["rows"] == 13 * 17_520

    exp = splits.experience_ids(df, settings.data)
    assert set(exp.unique()) == set(range(1, 13))
