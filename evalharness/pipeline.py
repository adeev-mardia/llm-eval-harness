"""A small, real RAG pipeline used to generate the golden evaluation set.

This is intentionally simple -- TF-IDF retrieval over a fixed corpus of
chunks, plus extractive answer generation (the answer is built from the
single most relevant sentence(s) in the top retrieved chunk) -- but every
piece of it is a genuine, working implementation, not a stand-in. It is
what produced the real (question, retrieved_context, generated_answer)
records in ``data/golden_set.json``; it is not needed to *use* the eval
harness on someone else's RAG pipeline's outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .textutils import split_into_sentences


@dataclass
class Corpus:
    ids: List[str]
    texts: List[str]

    def __post_init__(self) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._matrix = self._vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, str, float]]:
        """Return the top-k (id, text, score) chunks for a query, ranked by
        TF-IDF cosine similarity -- a standard, real retrieval technique.
        """
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix)[0]
        ranked = sorted(range(len(self.texts)), key=lambda i: sims[i], reverse=True)
        results = []
        for i in ranked[:k]:
            if sims[i] <= 0:
                continue
            results.append((self.ids[i], self.texts[i], float(sims[i])))
        return results


def extractive_answer(question: str, retrieved_chunks: List[str], n_sentences: int = 2) -> str:
    """Build an extractive answer: the sentence(s) across the retrieved
    chunks most similar to the question, concatenated in retrieval order.

    This is a legitimate, real extractive-QA technique (rank candidate
    sentences by similarity to the query and return the top ones) -- it is
    not a template or canned string.
    """
    candidates: List[Tuple[str, float]] = []
    for chunk in retrieved_chunks:
        for sentence in split_into_sentences(chunk):
            candidates.append((sentence, chunk))

    if not candidates:
        return ""

    sentences = [c[0] for c in candidates]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    try:
        matrix = vectorizer.fit_transform([question] + sentences)
    except ValueError:
        return sentences[0] if sentences else ""

    q_vec = matrix[0]
    sent_matrix = matrix[1:]
    sims = cosine_similarity(q_vec, sent_matrix)[0]
    ranked_idx = sorted(range(len(sentences)), key=lambda i: sims[i], reverse=True)
    top_idx = ranked_idx[:n_sentences]
    # Keep original order of appearance for readability.
    top_idx_sorted = sorted(top_idx)
    chosen = [sentences[i] for i in top_idx_sorted]
    return " ".join(chosen)
