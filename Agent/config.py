"""Load Agent/configs/agent.yaml (single source of defaults) with small environment overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

AGENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENT_DIR.parent
DEFAULT_CONFIG = AGENT_DIR / "configs" / "agent.yaml"


def load_config(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(p.read_text()) or {}
    cfg = copy.deepcopy(cfg)
    if os.environ.get("OLLAMA_HOST"):
        cfg["llm"]["base_url"] = os.environ["OLLAMA_HOST"]
    if os.environ.get("OLLAMA_MODEL"):
        cfg["llm"]["model"] = os.environ["OLLAMA_MODEL"]
    return cfg


def resolve(rel: Union[str, Path], root: Path = REPO_ROOT) -> Path:
    """Resolve a config path against the repo root unless it is already absolute."""
    p = Path(rel)
    return p if p.is_absolute() else root / p
