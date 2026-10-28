"""Day 1 pipeline: load FD001, validate, derive labels, split by engine, write reports.

Usage: python -m pipelines.prepare_data
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.data_loader import load_raw_history, load_test_rul
from src.labels import add_training_rul, build_test_endpoint_labels
from src.splits import save_manifest, split_engines
from src.validation import validate_raw

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())


def main() -> None:
    raw = Path(CFG["raw_dir"])
    out = Path(CFG["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    train = load_raw_history(raw / CFG["files"]["train"])
    test = load_raw_history(raw / CFG["files"]["test"])
    rul = load_test_rul(raw / CFG["files"]["test_rul"])

    report = {
        "dataset_id": CFG["dataset_id"],
        "schema_version": CFG["schema_version"],
        "train": validate_raw(train),
        "test": validate_raw(test),
        "n_test_rul_labels": int(len(rul)),
    }
    if report["test"]["n_engines"] != len(rul):
        raise SystemExit("Number of test engines does not match number of RUL labels")

    train = add_training_rul(train)
    ends = build_test_endpoint_labels(test, rul)
    manifest = split_engines(train["engine_id"].unique(), CFG["val_fraction"], CFG["seed"])

    save_manifest(manifest, out / "split_manifest.json")
    (out / "validation_report.json").write_text(json.dumps(report, indent=2))
    train.to_parquet(out / "train_labeled.parquet") if _has_parquet() else train.to_csv(out / "train_labeled.csv", index=False)
    ends.to_csv(out / "test_endpoints.csv", index=False)

    print(f"train: {report['train']['n_rows']} rows, {report['train']['n_engines']} engines")
    print(f"test:  {report['test']['n_rows']} rows, {report['test']['n_engines']} engines")
    print(f"split: {manifest['n_train_engines']} train / {manifest['n_val_engines']} validation engines (seed {manifest['seed']})")
    print(f"constant columns (candidates to drop): {report['train']['constant_columns']}")
    print(f"wrote reports to {out}/")


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
