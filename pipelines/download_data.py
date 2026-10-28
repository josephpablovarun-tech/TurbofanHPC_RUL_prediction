"""Download NASA C-MAPSS, keep FD001 only, record checksums (FRD: Source and schema, ML01).

Usage: python -m pipelines.download_data [--zip path/to/downloaded.zip]
If the download URL fails, download the archive manually from the NASA PCoE data repository
and pass it with --zip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import yaml

CFG = yaml.safe_load(Path("configs/data.yaml").read_text())


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, help="Use an already-downloaded archive")
    args = ap.parse_args()

    raw_dir = Path(CFG["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    zpath = args.zip or raw_dir.parent / "cmapss_archive.zip"
    if not args.zip:
        print(f"Downloading {CFG['download_url']}")
        urllib.request.urlretrieve(CFG["download_url"], zpath)

    wanted = set(CFG["files"].values())
    with zipfile.ZipFile(zpath) as z:
        names = {Path(n).name: n for n in z.namelist()}
        # Keep the dataset documentation alongside the data
        keep = wanted | {n for n in names if n.lower().startswith("readme")}
        for base in keep:
            if base in names:
                with z.open(names[base]) as src, (raw_dir / base).open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        missing = wanted - set(names)
        if missing:
            raise SystemExit(f"Archive is missing expected files: {sorted(missing)}")

    record = {
        "dataset_id": CFG["dataset_id"],
        "acquired": date.today().isoformat(),
        "parser_version": CFG["parser_version"],
        "source_url": CFG["download_url"],
        "sha256": {p.name: sha256(p) for p in sorted(raw_dir.iterdir()) if p.is_file() and p.name != "checksums.json"},
    }
    (raw_dir / "checksums.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))
    print("\nNext: read the README in", raw_dir, "and confirm column mapping and row counts before trusting them.")


if __name__ == "__main__":
    main()
