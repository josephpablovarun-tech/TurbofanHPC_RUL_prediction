import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.schema import RAW_COLUMNS


def make_history(engine_cycles: dict[int, int]) -> pd.DataFrame:
    """Build a small valid history: {engine_id: n_cycles}."""
    rows = []
    for eid, n in engine_cycles.items():
        for c in range(1, n + 1):
            rows.append([eid, c] + [0.1 * c + k for k in range(len(RAW_COLUMNS) - 2)])
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


@pytest.fixture
def history():
    return make_history({1: 5, 2: 8})
