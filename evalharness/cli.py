"""Command-line entry point for running the eval suite.

Usage:
    python -m evalharness.cli --dataset data/golden_set.json
    python -m evalharness.cli --dataset data/golden_set.json --threshold 0.55 --fail-under
    python -m evalharness.cli --dataset data/golden_set.json --worst 5
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .judge import HeuristicJudge
from .report import ContextChunk, EvalRecord, evaluate_dataset


def load_records(path: str) -> List[EvalRecord]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = []
    for item in raw:
        chunks = [ContextChunk(id=c["id"], text=c["text"]) for c in item["retrieved_context"]]
        records.append(
            EvalRecord(
                id=item["id"],
                question=item["question"],
                retrieved_context=chunks,
                answer=item["answer"],
                relevant_context_ids=item.get("relevant_context_ids"),
                reference_answer=item.get("reference_answer"),
            )
        )
    return records


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the offline RAG eval suite against a dataset.")
    parser.add_argument("--dataset", required=True, help="Path to a JSON eval dataset.")
    parser.add_argument("--threshold", type=float, default=None, help="Minimum aggregate score required.")
    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="Exit with status 1 if the aggregate score is below --threshold (for CI gating).",
    )
    parser.add_argument("--worst", type=int, default=5, help="Number of worst-performing records to print.")
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON instead of a summary.")
    args = parser.parse_args(argv)

    records = load_records(args.dataset)
    report = evaluate_dataset(records, judge=HeuristicJudge())

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Judge backend : {report.judge_name}")
        print(f"Records scored: {len(report.record_scores)}")
        print(f"Aggregate score         : {report.aggregate_score:.4f}")
        print(f"Mean faithfulness        : {report.mean_faithfulness:.4f}")
        print(f"Mean answer relevance    : {report.mean_relevance:.4f}")
        print(f"Mean context precision   : {report.mean_context_precision:.4f}")
        print(f"Mean context recall      : {report.mean_context_recall:.4f}")

        worst = report.worst(args.worst)
        if worst:
            print(f"\nWorst {len(worst)} record(s):")
            for s in worst:
                print(f"  [{s.record_id}] aggregate={s.aggregate:.4f} faithfulness={s.faithfulness.score:.4f} "
                      f"relevance={s.relevance_score:.4f}")
                print(f"    Q: {s.question}")
                unsupported = s.faithfulness.unsupported_claims
                if unsupported:
                    print("    Unsupported claim(s):")
                    for c in unsupported:
                        print(f"      - {c.text}")

    if args.threshold is not None and args.fail_under:
        if report.aggregate_score < args.threshold:
            print(
                f"\nFAIL: aggregate score {report.aggregate_score:.4f} is below threshold {args.threshold:.4f}",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: aggregate score {report.aggregate_score:.4f} meets threshold {args.threshold:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
