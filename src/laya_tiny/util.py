"""Small shared helpers: logging, hashing, JSONL I/O, seeding."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("laya_tiny")


def setup_logging(verbose: bool = False, log_file: Path | None = None) -> None:
    fmt = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
    for noisy in ("urllib3", "filelock", "huggingface_hub", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def text_id(text: str) -> str:
    """Content address of a text; identical texts share labels across runs."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Atomic write: a crash never leaves a half-written file that a later stage would trust."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
