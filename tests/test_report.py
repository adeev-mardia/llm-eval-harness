from evalharness.report import ContextChunk, EvalRecord, evaluate_dataset


def _make_record(record_id, question, answer, context_pairs, relevant_ids=None):
    chunks = [ContextChunk(id=cid, text=text) for cid, text in context_pairs]
    return EvalRecord(
        id=record_id,
        question=question,
        retrieved_context=chunks,
        answer=answer,
        relevant_context_ids=relevant_ids,
    )


def test_evaluate_dataset_computes_all_metrics_when_labels_present():
    record = _make_record(
        "r1",
        "What is Mercury's atmosphere like?",
        "Mercury has almost no atmosphere.",
        [("c1", "Mercury has almost no atmosphere and extreme temperature swings.")],
        relevant_ids=["c1"],
    )
    report = evaluate_dataset([record])
    assert len(report.record_scores) == 1
    score = report.record_scores[0]
    assert score.faithfulness.score == 1.0
    assert score.context is not None
    assert score.context.precision == 1.0
    assert score.context.recall == 1.0
    assert report.judge_name == "heuristic-tfidf"


def test_evaluate_dataset_skips_context_metrics_without_labels():
    record = _make_record(
        "r1",
        "What is Mercury's atmosphere like?",
        "Mercury has almost no atmosphere.",
        [("c1", "Mercury has almost no atmosphere.")],
        relevant_ids=None,
    )
    report = evaluate_dataset([record])
    score = report.record_scores[0]
    assert score.context is None
    # Aggregate should be the mean of just faithfulness + relevance.
    assert score.aggregate == (score.faithfulness.score + score.relevance_score) / 2


def test_report_aggregate_and_means_are_computed_correctly():
    good = _make_record(
        "good",
        "What is Mercury's atmosphere like?",
        "Mercury has almost no atmosphere.",
        [("c1", "Mercury has almost no atmosphere and swings in temperature.")],
        relevant_ids=["c1"],
    )
    bad = _make_record(
        "bad",
        "What is Mercury's atmosphere like?",
        "Mercury has seventeen rings made of solid gold.",
        [("c1", "Mercury has almost no atmosphere and swings in temperature.")],
        relevant_ids=["c1"],
    )
    report = evaluate_dataset([good, bad])
    assert report.mean_faithfulness < 1.0
    assert report.aggregate_score == round(
        sum(s.aggregate for s in report.record_scores) / 2, 4
    )


def test_worst_returns_lowest_aggregate_records_in_order():
    good = _make_record(
        "good",
        "What is Mercury's atmosphere like?",
        "Mercury has almost no atmosphere.",
        [("c1", "Mercury has almost no atmosphere and swings in temperature.")],
        relevant_ids=["c1"],
    )
    bad = _make_record(
        "bad",
        "What is Mercury's atmosphere like?",
        "Giraffes have very long necks for reaching tall trees.",
        [("c1", "Mercury has almost no atmosphere and swings in temperature.")],
        relevant_ids=["c1"],
    )
    report = evaluate_dataset([good, bad])
    worst = report.worst(1)
    assert worst[0].record_id == "bad"


def test_to_dict_contains_expected_keys_and_unsupported_claims():
    record = _make_record(
        "r1",
        "What is Mercury's atmosphere like?",
        "Mercury has almost no atmosphere. Mercury was visited by aliens in 1950.",
        [("c1", "Mercury has almost no atmosphere.")],
        relevant_ids=["c1"],
    )
    report = evaluate_dataset([record])
    d = report.to_dict()
    assert d["n_records"] == 1
    assert "aggregate_score" in d
    rec = d["records"][0]
    assert rec["unsupported_claims"] == ["Mercury was visited by aliens in 1950."]


def test_empty_dataset_returns_zeroed_report():
    report = evaluate_dataset([])
    assert report.record_scores == []
    assert report.aggregate_score == 0.0
    assert report.mean_faithfulness == 0.0
