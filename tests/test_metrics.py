from evalharness.metrics import (
    answer_relevance_score,
    context_precision_recall,
    faithfulness_score,
    split_into_claims,
)


def test_split_into_claims_splits_on_sentence_boundaries():
    text = "Mercury is the smallest planet. It has almost no atmosphere."
    claims = split_into_claims(text)
    assert claims == [
        "Mercury is the smallest planet.",
        "It has almost no atmosphere.",
    ]


def test_split_into_claims_handles_empty_string():
    assert split_into_claims("") == []
    assert split_into_claims("   ") == []


def test_faithfulness_fully_supported_answer_scores_high():
    context = [
        "Mercury is the closest planet to the Sun. It has almost no atmosphere, "
        "so temperatures swing from very hot in the day to very cold at night."
    ]
    answer = "Mercury is the closest planet to the Sun. It has almost no atmosphere."
    result = faithfulness_score(answer, context)
    assert result.score == 1.0
    assert all(c.supported for c in result.claims)
    assert result.unsupported_claims == []


def test_faithfulness_fabricated_claim_scores_low():
    context = [
        "Mercury is the closest planet to the Sun and the smallest planet in the Solar System."
    ]
    answer = (
        "Mercury is the closest planet to the Sun. "
        "Mercury has four small moons that were discovered by the Juno probe in 2015."
    )
    result = faithfulness_score(answer, context)
    # Exactly one of the two claims is genuinely fabricated and unsupported.
    assert result.score == 0.5
    unsupported_texts = [c.text for c in result.unsupported_claims]
    assert "Mercury has four small moons that were discovered by the Juno probe in 2015." in unsupported_texts


def test_faithfulness_with_no_context_is_zero():
    result = faithfulness_score("Mercury is the closest planet to the Sun.", [])
    assert result.score == 0.0
    assert len(result.unsupported_claims) == 1


def test_faithfulness_with_empty_answer_is_vacuously_one():
    result = faithfulness_score("", ["Some context."])
    assert result.score == 1.0
    assert result.claims == []


def test_faithfulness_entirely_unrelated_context_scores_zero():
    context = ["Blueberries are a good source of antioxidants and fiber."]
    answer = "The Great Wall of China stretches for thousands of kilometers across northern China."
    result = faithfulness_score(answer, context)
    assert result.score == 0.0


def test_answer_relevance_on_topic_scores_higher_than_off_topic():
    question = "Why is Venus the hottest planet in the Solar System?"
    on_topic_answer = (
        "Venus is the hottest planet because its thick carbon dioxide atmosphere traps heat."
    )
    off_topic_answer = (
        "Halley's Comet returns to the inner Solar System approximately every seventy-six years."
    )
    on_topic = answer_relevance_score(question, on_topic_answer)
    off_topic = answer_relevance_score(question, off_topic_answer)
    assert on_topic.score > off_topic.score


def test_answer_relevance_handles_empty_strings():
    assert answer_relevance_score("", "").score == 0.0
    assert answer_relevance_score("What is Mars?", "").score == 0.0


def test_context_precision_recall_perfect_match():
    result = context_precision_recall(["a", "b"], ["a", "b"])
    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0


def test_context_precision_recall_partial_match():
    # Retrieved 3 chunks, only 1 of which is actually relevant; the
    # relevant set has 2 total chunks, so recall is 1/2.
    result = context_precision_recall(["a", "b", "c"], ["a", "x"])
    assert result.precision == round(1 / 3, 4)
    assert result.recall == 0.5


def test_context_precision_recall_no_overlap():
    result = context_precision_recall(["a", "b"], ["x", "y"])
    assert result.precision == 0.0
    assert result.recall == 0.0
    assert result.f1 == 0.0


def test_context_precision_recall_empty_retrieved():
    result = context_precision_recall([], ["a"])
    assert result.precision == 0.0
    assert result.recall == 0.0


def test_context_precision_recall_dedupes_retrieved_ids():
    result = context_precision_recall(["a", "a", "b"], ["a"])
    assert result.retrieved_ids == ["a", "b"]
    assert result.precision == 0.5
