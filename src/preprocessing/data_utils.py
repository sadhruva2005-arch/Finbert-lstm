"""
Shared DataFrame utilities: column detection, date normalisation,
label building, and train/val/test splitting.
"""

from __future__ import annotations

import glob

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Column detection helpers
# ──────────────────────────────────────────────

_DATE_CANDIDATES = ["date", "published", "Date", "Published"]
_TEXT_CANDIDATES = ["article", "summary", "headline", "text", "content", "body", "Title", "title"]
_LABEL_CANDIDATES = ["label", "sentiment", "target"]

_POS_WORDS = {
    "gain", "surge", "upbeat", "profit", "beat", "strong", "growth", "rally",
    "bullish", "upgrade", "outperform", "record", "robust", "improve", "higher",
    "boost", "optimism",
}
_NEG_WORDS = {
    "loss", "slump", "downgrade", "weak", "miss", "decline", "risk", "bearish",
    "fall", "plunge", "cut", "shortfall", "slowdown", "concern", "lower", "drop",
    "crisis", "volatility",
}


def pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name (exact, then case-insensitive)."""
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_dataframe(path: str | None = None) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV path, or auto-discover the first CSV
    in /kaggle/input or the current directory.
    """
    if path:
        return pd.read_csv(path)
    paths = glob.glob("/kaggle/input/**/*.csv", recursive=True) or glob.glob("*.csv")
    if not paths:
        raise FileNotFoundError(
            "No CSV found. Provide an explicit path or place a CSV in the working directory."
        )
    return pd.read_csv(paths[0])


def normalise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """
    Detect and normalise the date and text columns.

    Returns:
        (df_normalised, date_col_name, text_col_name)
    """
    df = df.copy()

    date_col = pick_column(df, _DATE_CANDIDATES)
    text_col = pick_column(df, _TEXT_CANDIDATES)

    if date_col is None:
        df["date"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        date_col = "date"
        print("[DataUtils] No date column found; created placeholder.")
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")

    if text_col is None:
        raise ValueError(
            "No text column found. Expected one of: "
            + ", ".join(_TEXT_CANDIDATES)
        )

    df[text_col] = df[text_col].astype(str).fillna("")
    if text_col not in {"article", "summary", "headline", "text"}:
        df["text"] = df[text_col]
        text_col = "text"

    return df, date_col, text_col


def build_labels(df: pd.DataFrame, text_col: str) -> pd.Series:
    """
    Return binary labels (0/1).  Uses df['label'] if present and valid,
    otherwise falls back to a finance lexicon weak-labelling.
    """
    label_col = pick_column(df, _LABEL_CANDIDATES)
    if label_col and df[label_col].notna().any():
        def _to01(x):
            s = str(x).strip().lower()
            if s in {"1", "pos", "positive"}:
                return 1
            if s in {"0", "neg", "negative"}:
                return 0
            try:
                return 1 if int(float(s)) == 1 else 0
            except Exception:
                return np.nan

        y = df[label_col].map(_to01)
        if y.dropna().nunique() >= 2:
            return y.fillna(0).astype(int)

    def lex_score(s: str) -> int:
        words = s.lower().split()
        return sum(w in _POS_WORDS for w in words) - sum(w in _NEG_WORDS for w in words)

    return (df[text_col].map(lex_score) >= 0).astype(int)


def temporal_split(
    df: pd.DataFrame,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Order-preserving (temporal) train / validation / test split.

    Returns:
        (train_df, val_df, test_df)
    """
    n = len(df)
    test_n = int(n * test_frac)
    val_n = int((n - test_n) * val_frac)
    train_n = n - test_n - val_n

    return (
        df.iloc[:train_n].copy(),
        df.iloc[train_n : train_n + val_n].copy(),
        df.iloc[train_n + val_n :].copy(),
    )
