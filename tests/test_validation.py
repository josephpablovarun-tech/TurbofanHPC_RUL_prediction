"""AT01: duplicate cycle, NaN sensor and missing column are each rejected with a specific error."""
import numpy as np
import pytest

from src.validation import DataValidationError, find_cycle_gaps, validate_raw


def test_valid_history_passes(history):
    report = validate_raw(history)
    assert report["n_engines"] == 2 and report["n_rows"] == 13


def test_missing_column_rejected(history):
    with pytest.raises(DataValidationError, match="sensor_21"):
        validate_raw(history.drop(columns=["sensor_21"]))


def test_nan_sensor_rejected(history):
    history.loc[3, "sensor_4"] = np.nan
    with pytest.raises(DataValidationError, match="sensor_4"):
        validate_raw(history)


def test_duplicate_key_rejected(history):
    dup = history.copy()
    dup.loc[1, "cycle"] = 1
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_raw(dup)


def test_cycle_order_rejected(history):
    bad = history.iloc[[1, 0] + list(range(2, len(history)))].reset_index(drop=True)
    with pytest.raises(DataValidationError, match="strictly increasing"):
        validate_raw(bad)


def test_cycle_gap_is_warning_not_error(history):
    gappy = history.drop(index=2).reset_index(drop=True)
    assert find_cycle_gaps(gappy) == {1: [2]}
    assert validate_raw(gappy)["warnings"]
