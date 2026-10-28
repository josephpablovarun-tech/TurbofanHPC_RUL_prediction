"""Read-only access to the FD001 train and test histories, with selection checks.

Engine ids are unique only within a split, so every lookup takes the split (FRD data schema).
"""
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from src.data_loader import load_raw_history, load_test_rul


class SelectionError(ValueError):
    """The requested split, engine or cutoff is not valid for the loaded data."""


class DataStore:
    def __init__(self, raw_dir: Union[str, Path], train_file: str, test_file: str, rul_file: Optional[str] = None):
        raw_dir = Path(raw_dir)
        self._paths = {"train": raw_dir / train_file, "test": raw_dir / test_file}
        self._rul_path: Optional[Path] = (raw_dir / rul_file) if rul_file else None
        self._frames: Dict[str, pd.DataFrame] = {}
        self._rul: Optional[pd.Series] = None
        self._checksum = None

    def _frame(self, split: str) -> pd.DataFrame:
        if split not in self._paths:
            raise SelectionError(f"Unknown dataset split '{split}' (use 'train' or 'test').")
        if split not in self._frames:
            path = self._paths[split]
            if not path.exists():
                raise SelectionError(f"Data file not found: {path.name}")
            df = load_raw_history(path)
            self._frames[split] = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
        return self._frames[split]

    def frame(self, split: str) -> pd.DataFrame:
        """Whole split frame for reference statistics (never exposed to the LLM or to inference tools)."""
        return self._frame(split)

    @property
    def data_checksum(self) -> str:
        if self._checksum is None:
            h = hashlib.sha256()
            for split in ("train", "test"):
                if self._paths[split].exists():
                    h.update(self._paths[split].read_bytes())
            self._checksum = h.hexdigest()[:12]
        return self._checksum

    def engine_ids(self, split: str) -> List[int]:
        return sorted(int(i) for i in self._frame(split)["engine_id"].unique())

    def max_cycle(self, split: str, engine_id: int) -> int:
        df = self._frame(split)
        rows = df.loc[df["engine_id"] == engine_id, "cycle"]
        if rows.empty:
            raise SelectionError(f"Engine {engine_id} does not exist in the {split} split.")
        return int(rows.max())

    def validate_selection(self, split: str, engine_id: int, cutoff: int) -> int:
        """Return the engine's last observed cycle, or raise ``SelectionError``."""
        last = self.max_cycle(split, engine_id)
        if cutoff < 1 or cutoff > last:
            raise SelectionError(
                f"Cutoff cycle {cutoff} is outside the observed history of {split} engine {engine_id} (cycles 1 to {last}). "
                "No later cycles are available, so none can be used."
            )
        return last

    # ------------------------------------------------------------------ ground truth (evaluation display only)
    def rul_labels(self) -> Optional[pd.Series]:
        """Official true RUL of each TEST engine at its last observed cycle, indexed by engine id; None if the label file is absent.

        Evaluation and data exploration only: the agent tools and the chat never call this, so no label reaches inference.
        """
        if self._rul_path is None or not self._rul_path.exists():
            return None
        if self._rul is None:
            s = load_test_rul(self._rul_path)
            ids = self.engine_ids("test")
            if list(s.index) != ids:
                raise SelectionError(f"{self._rul_path.name} has {len(s)} labels but the test split has {len(ids)} engines (ids must align).")
            self._rul = s
        return self._rul

    def true_rul_at(self, split: str, engine_id: int, cutoff: int) -> Optional[int]:
        """True remaining life at ``cutoff``: last cycle minus cutoff for a train engine; label plus (last cycle - cutoff) for a test engine."""
        last = self.max_cycle(split, engine_id)
        if split == "train":
            return last - cutoff
        labels = self.rul_labels()
        if labels is None or engine_id not in labels.index:
            return None
        return int(labels.loc[engine_id]) + (last - cutoff)

    def full_history(self, split: str, engine_id: int) -> pd.DataFrame:
        """Every row of a TRAIN engine (its complete run to failure) for retrospective analysis only.

        Refused for the test split: test histories are truncated, and nothing after the selected cutoff may ever be read there.
        """
        if split != "train":
            raise SelectionError("A full run to failure is only available for train engines. Test histories stop before failure, "
                                 "and no cycle after the cutoff is ever read for them.")
        self.max_cycle(split, engine_id)  # raises SelectionError for an unknown engine
        df = self._frame(split)
        return df.loc[df["engine_id"] == engine_id].reset_index(drop=True)

    def history(self, split: str, engine_id: int, cutoff: int) -> pd.DataFrame:
        """Observed rows with cycle <= cutoff (later cycles never leave the store)."""
        self.validate_selection(split, engine_id, cutoff)
        df = self._frame(split)
        return df.loc[(df["engine_id"] == engine_id) & (df["cycle"] <= cutoff)].reset_index(drop=True)
