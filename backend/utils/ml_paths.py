"""Filesystem-safe names for saved models (avoids '^' in paths)."""


def artifact_stem(ticker: str) -> str:
    """Turn '^NDX' into '_NDX' for filenames while staying reversible in our convention."""
    return ticker.strip().replace("^", "_").replace("/", "_")
