"""Record model and report builder: runs the full metric suite over a set
of (question, context, answer) records and produces a scored report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional, Sequence

from .judge import HeuristicJudge, Judge
from .metrics import ContextScoreResult, FaithfulnessResult, context_precision_recall


@dataclass
class ContextChunk:
    id: str
    text: str


@dataclass
class EvalRecord:
    """One RAG pipeline output to score.

    ``retrieved_context`` is the ordered list of context chunks the
    retriever actually returned. ``relevant_context_ids`` is the
    hand-labeled ground truth: the ids of chunks that are genuinely
    relevant to ``question`` (used for context precision/recall). When
    omitted, context precision/recall is skipped for that record.
    """

    id: str
    question: str
    retrieved_context: List[ContextChunk]
    answer: str
    relevant_context_ids: Optional[List[str]] = None
    reference_answer: Optional[str] = None

    @property
    def context_texts(self) -> List[str]:
        return [c.text for c in self.retrieved_context]


@dataclass
class RecordScore:
    record_id: str
    question: str
    answer: str
    faithfulness: FaithfulnessResult
    relevance_score: float
    context: Optional[ContextScoreResult]

    @property
    def aggregate(self) -> float:
        """Mean of the metrics that were actually computed for this
        record (faithfulness and relevance always; context precision/
        recall only when ground-truth labels were supplied).
        """
        parts = [self.faithfulness.score, self.relevance_score]
        if self.context is not None:
            parts.append(self.context.f1)
        return sum(parts) / len(parts)


@dataclass
class EvalReport:
    record_scores: List[RecordScore]
    judge_name: str

    @property
    def mean_faithfulness(self) -> float:
        return round(mean(s.faithfulness.score for s in self.record_scores), 4) if self.record_scores else 0.0

    @property
    def mean_relevance(self) -> float:
        return round(mean(s.relevance_score for s in self.record_scores), 4) if self.record_scores else 0.0

    @property
    def mean_context_precision(self) -> float:
        scored = [s.context.precision for s in self.record_scores if s.context is not None]
        return round(mean(scored), 4) if scored else 0.0

    @property
    def mean_context_recall(self) -> float:
        scored = [s.context.recall for s in self.record_scores if s.context is not None]
        return round(mean(scored), 4) if scored else 0.0

    @property
    def aggregate_score(self) -> float:
        """Overall aggregate: mean of the per-record aggregates. This is
        the single number the CI gate compares against its threshold.
        """
        return round(mean(s.aggregate for s in self.record_scores), 4) if self.record_scores else 0.0

    def worst(self, n: int = 5) -> List[RecordScore]:
        return sorted(self.record_scores, key=lambda s: s.aggregate)[:n]

    def to_dict(self) -> Dict:
        return {
            "judge": self.judge_name,
            "n_records": len(self.record_scores),
            "aggregate_score": self.aggregate_score,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_answer_relevance": self.mean_relevance,
            "mean_context_precision": self.mean_context_precision,
            "mean_context_recall": self.mean_context_recall,
            "records": [
                {
                    "id": s.record_id,
                    "question": s.question,
                    "answer": s.answer,
                    "faithfulness": s.faithfulness.score,
                    "answer_relevance": s.relevance_score,
                    "context_precision": s.context.precision if s.context else None,
                    "context_recall": s.context.recall if s.context else None,
                    "aggregate": round(s.aggregate, 4),
                    "unsupported_claims": [c.text for c in s.faithfulness.unsupported_claims],
                }
                for s in self.record_scores
            ],
        }


def evaluate_dataset(records: Sequence[EvalRecord], judge: Optional[Judge] = None) -> EvalReport:
    """Run the full metric suite over a sequence of :class:`EvalRecord`.

    ``judge`` defaults to :class:`evalharness.judge.HeuristicJudge`, the
    fully offline TF-IDF / n-gram backend. Pass an
    :class:`evalharness.judge.OpenAICompatibleJudge` instance to score with
    a real hosted or local LLM instead.
    """
    judge = judge or HeuristicJudge()
    scores: List[RecordScore] = []
    for record in records:
        context_texts = record.context_texts
        faithfulness = judge.score_faithfulness(record.answer, context_texts)
        relevance = judge.score_relevance(record.question, record.answer)

        context_result: Optional[ContextScoreResult] = None
        if record.relevant_context_ids is not None:
            retrieved_ids = [c.id for c in record.retrieved_context]
            context_result = context_precision_recall(retrieved_ids, record.relevant_context_ids)

        scores.append(
            RecordScore(
                record_id=record.id,
                question=record.question,
                answer=record.answer,
                faithfulness=faithfulness,
                relevance_score=relevance.score,
                context=context_result,
            )
        )
    return EvalReport(record_scores=scores, judge_name=judge.name)
