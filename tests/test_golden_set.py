"""Sanity checks on the shipped golden evaluation set and the CI gate's
real behavior against it. This is the test the CI workflow's eval-gate
step depends on conceptually (the workflow re-runs the CLI directly, but
this test keeps the dataset itself honest under pytest too).
"""

import json
import os

from evalharness.cli import load_records
from evalharness.judge import HeuristicJudge
from evalharness.report import evaluate_dataset

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "golden_set.json")

# Documented CI gate threshold -- see README and .github/workflows/eval-gate.yml.
CI_GATE_THRESHOLD = 0.50


def test_golden_set_file_is_well_formed_and_sizeable():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    assert len(raw) >= 25
    for item in raw:
        assert item["id"]
        assert item["question"]
        assert item["answer"]
        assert len(item["retrieved_context"]) > 0
        assert item["relevant_context_ids"]


def test_golden_set_contains_both_clean_and_adversarial_cases():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    case_types = {item["case_type"] for item in raw}
    assert "clean" in case_types
    assert "fabricated" in case_types
    assert "off_topic" in case_types


def test_golden_set_passes_the_documented_ci_gate_threshold():
    records = load_records(GOLDEN_SET_PATH)
    report = evaluate_dataset(records, judge=HeuristicJudge())
    assert report.aggregate_score >= CI_GATE_THRESHOLD


def test_fabricated_cases_have_lower_faithfulness_than_clean_cases():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = load_records(GOLDEN_SET_PATH)
    report = evaluate_dataset(records, judge=HeuristicJudge())
    by_id = {s.record_id: s for s in report.record_scores}
    case_type_by_id = {item["id"]: item["case_type"] for item in raw}

    clean_scores = [by_id[rid].faithfulness.score for rid, t in case_type_by_id.items() if t == "clean"]
    fabricated_scores = [
        by_id[rid].faithfulness.score for rid, t in case_type_by_id.items() if t == "fabricated"
    ]
    assert sum(fabricated_scores) / len(fabricated_scores) < sum(clean_scores) / len(clean_scores)


def test_off_topic_cases_have_lower_context_precision_than_clean_cases():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    records = load_records(GOLDEN_SET_PATH)
    report = evaluate_dataset(records, judge=HeuristicJudge())
    by_id = {s.record_id: s for s in report.record_scores}
    case_type_by_id = {item["id"]: item["case_type"] for item in raw}

    clean_precision = [
        by_id[rid].context.precision for rid, t in case_type_by_id.items() if t == "clean"
    ]
    off_topic_precision = [
        by_id[rid].context.precision for rid, t in case_type_by_id.items() if t == "off_topic"
    ]
    assert sum(off_topic_precision) / len(off_topic_precision) < sum(clean_precision) / len(clean_precision)
