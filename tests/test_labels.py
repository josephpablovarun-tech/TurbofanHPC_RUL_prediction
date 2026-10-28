"""AT02: target derivation."""
import pandas as pd

from src.labels import add_test_trajectory_rul, add_training_rul, build_test_endpoint_labels
from tests.conftest import make_history


def test_training_rul_fixture_ending_cycle_5():
    df = add_training_rul(make_history({1: 5}))
    assert df.loc[df.cycle == 1, "rul_cycles"].item() == 4
    assert df.loc[df.cycle == 5, "rul_cycles"].item() == 0


def test_training_rul_is_per_engine():
    df = add_training_rul(make_history({1: 3, 2: 6}))
    assert df[(df.engine_id == 1) & (df.cycle == 3)].rul_cycles.item() == 0
    assert df[(df.engine_id == 2) & (df.cycle == 3)].rul_cycles.item() == 3


def test_test_endpoint_keeps_supplied_rul():
    test = make_history({1: 5})
    rul = pd.Series([12], index=pd.Index([1], name="engine_id"), name="rul_cycles")
    ends = build_test_endpoint_labels(test, rul)
    assert ends.loc[0, "last_cycle"] == 5 and ends.loc[0, "rul_cycles"] == 12


def test_test_trajectory_truth():
    test = make_history({1: 5})
    rul = pd.Series([12], index=pd.Index([1], name="engine_id"), name="rul_cycles")
    traj = add_test_trajectory_rul(test, rul)
    assert traj[traj.cycle == 5].rul_cycles.item() == 12
    assert traj[traj.cycle == 1].rul_cycles.item() == 16
