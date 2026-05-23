"""
NLP text preprocessing pipeline: lowercasing, stopword removal,
punctuation cleaning, number removal, lemmatization, POS tagging, chunking.
"""

import string

import nltk
import numpy as np
from nltk import FreqDist, pos_tag
from nltk.corpus import stopwords
from nltk.stem import wordnet
from nltk.tokenize import word_tokenize

# Download required NLTK data (safe to call multiple times)
for _pkg in ("stopwords", "punkt", "wordnet", "averaged_perceptron_tagger", "tagsets_json"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_lemmatizer = wordnet.WordNetLemmatizer()


# ──────────────────────────────────────────────
# Individual transformation functions
# ──────────────────────────────────────────────

def lowercase(text: str) -> str:
    return text.lower()


def replace_percent(text: str) -> str:
    return text.replace("%", " percent")


def remove_punctuations(text: str) -> str:
    translator = str.maketrans("", "", string.punctuation)
    return text.strip().translate(translator)


def remove_stop_words(text: str) -> list[str]:
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return [w.strip() for w in tokens if w not in stop_words and w.strip()]


def remove_numbers(tokens: list[str]) -> list[str]:
    return ["" if t.isdigit() else t for t in tokens]


def drop_empty_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t]


def lemmatize(tokens: list[str]) -> list[str]:
    return [_lemmatizer.lemmatize(w, pos="v") for w in tokens]


def pos_tagg(tokens: list[str]):
    return pos_tag(tokens)


def chunk(tagged_tokens) -> nltk.Tree:
    grammar = "NP: {<DT>?<JJ>*<NN>}"
    parser = nltk.RegexpParser(grammar)
    return parser.parse(tagged_tokens)


def word_frequency(tokens: list[str]) -> FreqDist:
    tokens = [t for t in tokens if t]
    return FreqDist(tokens)


# ──────────────────────────────────────────────
# Full pipeline (applies all steps to a column)
# ──────────────────────────────────────────────

def preprocess_column(series, include_pos: bool = False, include_chunk: bool = False):
    """
    Apply the full NLP preprocessing pipeline to a pandas Series of strings.

    Args:
        series: pd.Series of raw text.
        include_pos: If True, apply POS tagging as the final step.
        include_chunk: If True, apply chunking after POS tagging.

    Returns:
        Transformed pd.Series.
    """
    s = series.astype(str).fillna("")
    s = s.apply(lowercase)
    s = s.apply(replace_percent)
    s = s.apply(remove_punctuations)
    s = s.apply(remove_stop_words)
    s = s.apply(remove_numbers)
    s = s.apply(drop_empty_tokens)
    s = s.apply(lemmatize)
    s = s.apply(drop_empty_tokens)

    if include_pos or include_chunk:
        s = s.apply(pos_tagg)

    if include_chunk:
        s = s.apply(chunk)

    return s
