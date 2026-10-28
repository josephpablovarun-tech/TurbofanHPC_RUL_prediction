"""Loader handles trailing empty columns and rejects unexpected content."""
import pytest

from src.data_loader import load_raw_history
from src.schema import RAW_COLUMNS
from src.validation import DataValidationError, validate_raw


def _line(eid, c, extra=""):
    vals = " ".join(str(round(0.1 * c + k, 4)) for k in range(24))
    return f"{eid} {c} {vals}{extra}"


def test_trailing_space_columns_removed(tmp_path):
    p = tmp_path / "train.txt"
    p.write_text("\n".join(_line(1, c, "  ") for c in range(1, 4)) + "\n")
    df = load_raw_history(p)
    assert list(df.columns) == RAW_COLUMNS
    assert validate_raw(df)["n_rows"] == 3


def test_unexpected_nonempty_column_rejected(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("\n".join(_line(1, c, " 9.9") for c in range(1, 4)) + "\n")
    with pytest.raises(DataValidationError, match="unexpected"):
        load_raw_history(p)
