"""
evalharness: an offline-first evaluation harness for RAG / LLM pipeline outputs.

The package computes faithfulness, answer relevance, and context precision/recall
scores for (question, retrieved_context, generated_answer) records using
genuine text-similarity techniques (TF-IDF cosine similarity and lexical
n-gram overlap), with no dependency on a hosted model. An optional LLM-judge
backend can be swapped in when an API key or local inference server is
available.
"""

from .metrics import (
    Claim,
    FaithfulnessResult,
    RelevanceResult,
    ContextScoreResult,
    split_into_claims,
    faithfulness_score,
    answer_relevance_score,
    context_precision_recall,
)
from .report import EvalRecord, EvalReport, evaluate_dataset
from .judge import HeuristicJudge, OpenAICompatibleJudge, Judge

__all__ = [
    "Claim",
    "FaithfulnessResult",
    "RelevanceResult",
    "ContextScoreResult",
    "split_into_claims",
    "faithfulness_score",
    "answer_relevance_score",
    "context_precision_recall",
    "EvalRecord",
    "EvalReport",
    "evaluate_dataset",
    "HeuristicJudge",
    "OpenAICompatibleJudge",
    "Judge",
]

__version__ = "0.1.0"
