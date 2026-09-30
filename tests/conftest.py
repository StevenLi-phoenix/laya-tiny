from __future__ import annotations

import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def smoke_cfg():
    from laya_tiny.config import load_config

    return load_config(ROOT / "configs", "smoke")


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A throwaway project root with the real configs and holdout."""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    (tmp_path / "data" / "holdout").mkdir(parents=True)
    shutil.copy2(ROOT / "data" / "holdout" / "tickets.jsonl", tmp_path / "data" / "holdout" / "tickets.jsonl")
    return tmp_path
