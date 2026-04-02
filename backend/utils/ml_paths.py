"""
Per-ticker model directories under MODEL_DIR.

Layout: MODEL_DIR/<ticker_dir>/lstm.keras, lstm_meta.joblib, arima.pkl
(^NDX uses folder NDX; slashes in symbols map to underscores.)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def artifact_stem(ticker: str) -> str:
    """
    Sanitized single path segment for the ticker's model subfolder.

    Strips '^' (e.g. ^NDX -> NDX). Replaces '/' with '_' so the name is one directory level.
    """
    s = ticker.strip().replace("/", "_").replace("^", "")
    return s if s else "UNKNOWN"


def _legacy_flat_filename_stem_to_dir_name(flat_stem: str) -> str:
    """
    Map a legacy root-level filename stem to the new per-ticker folder name.

    Old artifact_stem replaced '^' with '_', so ^NDX became _NDX in *_lstm.keras; new folder is NDX.
    """
    if flat_stem.startswith("_"):
        return flat_stem[1:]
    return flat_stem


def migrate_flat_to_subfolders(model_dir: Path | None = None) -> None:
    """
    Move legacy flat files into per-ticker subfolders (idempotent).

    Scans MODEL_DIR root only for *_lstm.keras, *_lstm_meta.joblib, *_arima.pkl.
    Skips anything already inside a subdirectory. Skips a move if the destination file exists.
    Logs each successful move as: Migrated <oldname> → <dir>/<newname>
    """
    from config import settings

    root = model_dir if model_dir is not None else settings.MODEL_DIR
    if not root.is_dir():
        return

    for entry in list(root.iterdir()):
        # Skip nested packages (already migrated or unrelated dirs).
        if entry.is_dir():
            continue
        if not entry.is_file():
            continue
        name = entry.name
        dest_basename: str | None = None
        legacy_stem: str | None = None
        if name.endswith("_lstm.keras"):
            legacy_stem = name[: -len("_lstm.keras")]
            dest_basename = "lstm.keras"
        elif name.endswith("_lstm_meta.joblib"):
            legacy_stem = name[: -len("_lstm_meta.joblib")]
            dest_basename = "lstm_meta.joblib"
        elif name.endswith("_arima.pkl"):
            legacy_stem = name[: -len("_arima.pkl")]
            dest_basename = "arima.pkl"
        else:
            continue

        if not legacy_stem:
            logger.warning("Skipping malformed legacy artifact name: %s", name)
            continue

        dir_name = _legacy_flat_filename_stem_to_dir_name(legacy_stem)
        dest_dir = root / dir_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / dest_basename
        if dest.exists():
            logger.warning("Migration skip (target exists): %s -> %s", entry, dest)
            continue
        entry.rename(dest)
        logger.info("Migrated %s → %s/%s", name, dir_name, dest_basename)
