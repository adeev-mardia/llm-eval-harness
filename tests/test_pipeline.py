from evalharness.pipeline import Corpus, extractive_answer


def test_corpus_retrieves_most_similar_chunk_first():
    corpus = Corpus(
        ids=["a", "b", "c"],
        texts=[
            "Mercury is the closest planet to the Sun.",
            "Bananas are a good source of potassium.",
            "Mercury has almost no atmosphere.",
        ],
    )
    results = corpus.retrieve("Tell me about Mercury's distance from the Sun", k=2)
    returned_ids = [r[0] for r in results]
    assert "a" in returned_ids
    assert "b" not in returned_ids  # unrelated chunk should not be a top-2 match


def test_corpus_retrieve_returns_empty_for_completely_unrelated_query():
    corpus = Corpus(ids=["a"], texts=["Mercury is the closest planet to the Sun."])
    results = corpus.retrieve("zzz qqq xyz nonsense gibberish", k=3)
    assert results == []


def test_extractive_answer_pulls_relevant_sentence():
    chunks = [
        "Mercury is the closest planet to the Sun. Bananas are yellow and curved.",
    ]
    answer = extractive_answer("What is the closest planet to the Sun?", chunks, n_sentences=1)
    assert "Mercury" in answer
    assert "Bananas" not in answer


def test_extractive_answer_handles_no_chunks():
    assert extractive_answer("Any question?", [], n_sentences=2) == ""
