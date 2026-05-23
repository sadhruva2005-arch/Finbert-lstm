"""
Custom TF-IDF implementation built from article tokens.
Produces a document-term matrix saved as TFIDF.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nltk import FreqDist


def token_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency (relative) for a list of tokens."""
    tokens = [t for t in tokens if t]
    if not tokens:
        return {}
    fd = FreqDist(tokens)
    return {word: count / len(tokens) for word, count in fd.items()}


def build_idf(article_series: pd.Series) -> dict[str, float]:
    """
    Compute IDF scores for all unique terms across the corpus.

    Args:
        article_series: Series of token lists (already preprocessed).

    Returns:
        dict mapping term -> IDF score.
    """
    # Build global vocabulary
    all_tokens: list[str] = []
    for tokens in article_series:
        all_tokens.extend([t for t in tokens if t])
    unique_words = set(all_tokens)

    n_docs = len(article_series)
    idf: dict[str, float] = {}
    for word in unique_words:
        doc_count = sum(1 for tokens in article_series if word in tokens)
        idf[word] = np.log(n_docs / max(doc_count, 1))
    return idf


def compute_tfidf(article_series: pd.Series) -> pd.DataFrame:
    """
    Compute TF-IDF matrix for the corpus.

    Args:
        article_series: Series of token lists (already preprocessed).

    Returns:
        pd.DataFrame of shape (n_docs, n_terms), filled with 0 for missing terms.
    """
    tf_per_doc = article_series.apply(token_frequency)
    idf = build_idf(article_series)

    tfidf_rows = []
    for tf in tf_per_doc:
        row = [tf.get(word, 0.0) * idf.get(word, 0.0) for word in idf]
        tfidf_rows.append(row)

    return pd.DataFrame(tfidf_rows, columns=list(idf.keys())).fillna(0)
