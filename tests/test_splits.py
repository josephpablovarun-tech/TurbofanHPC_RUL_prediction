"""AT01/AT03: split membership is recorded, disjoint and reproducible."""
from src.splits import load_manifest, save_manifest, split_engines


def test_split_disjoint_complete_and_sized():
    m = split_engines(range(1, 101), 0.2, 42)
    assert m["n_val_engines"] == 20 and m["n_train_engines"] == 80
    assert not set(m["train_engine_ids"]) & set(m["val_engine_ids"])
    assert set(m["train_engine_ids"]) | set(m["val_engine_ids"]) == set(range(1, 101))


def test_split_reproducible_and_seed_sensitive():
    assert split_engines(range(1, 101), 0.2, 42) == split_engines(range(1, 101), 0.2, 42)
    assert split_engines(range(1, 101), 0.2, 42) != split_engines(range(1, 101), 0.2, 7)


def test_manifest_roundtrip(tmp_path):
    m = split_engines(range(1, 11), 0.2, 1)
    save_manifest(m, tmp_path / "m.json")
    assert load_manifest(tmp_path / "m.json") == m
