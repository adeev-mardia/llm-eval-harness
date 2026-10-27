"""Small, dependency-light text utilities shared by the metrics module.

These are deliberately simple, inspectable implementations rather than
calls out to a heavyweight NLP toolkit -- the point of this project is
that every number it produces can be traced back to a concrete, readable
piece of logic.
"""

from __future__ import annotations

import re
from typing import Iterable, List

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")

# A small, fixed stopword list used only for n-gram overlap scoring (not for
# TF-IDF, which has its own statistically-derived weighting via IDF).
STOPWORDS = frozenset(
    """
    a an the this that these those is are was were be been being
    of to in on at for with by from as it its into over under
    and or but if then than so such not no nor
    i you he she we they them him her his hers our your their
    do does did doing have has had having
    can could will would should may might must shall
    what which who whom where when why how
    there here
    """.split()
)


def split_into_sentences(text: str) -> List[str]:
    """Split a block of text into sentence-like units.

    Uses punctuation boundaries followed by a capital letter / digit as the
    split point. This is a heuristic, not a full sentence tokenizer, but it
    is adequate for the short, well-punctuated answers this harness scores,
    and it has no external dependency.
    """
    text = text.strip()
    if not text:
        return []
    # Normalize whitespace first.
    text = re.sub(r"\s+", " ", text)
    parts = _SENTENCE_SPLIT_RE.split(text)
    # Final cleanup: drop empties, strip trailing/leading whitespace.
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used for n-gram overlap scoring."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def content_tokens(text: str) -> List[str]:
    """Tokens with stopwords removed, for overlap scoring."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def ngrams(tokens: Iterable[str], n: int) -> List[tuple]:
    tokens = list(tokens)
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def ngram_overlap_score(claim_tokens: List[str], source_tokens: List[str], n: int = 2) -> float:
    """Fraction of the claim's n-grams (falling back to unigrams when the
    claim is too short for n-grams) that also appear in the source tokens.
    """
    if not claim_tokens:
        return 0.0
    claim_ngrams = ngrams(claim_tokens, n) or ngrams(claim_tokens, 1)
    source_ngram_set = set(ngrams(source_tokens, n)) | set(ngrams(source_tokens, 1))
    if not claim_ngrams:
        return 0.0
    hits = sum(1 for g in claim_ngrams if g in source_ngram_set)
    return hits / len(claim_ngrams)
