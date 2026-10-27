"""Builds data/golden_set.json.

Runs the real TF-IDF retrieval + extractive-answer pipeline in
evalharness.pipeline against the corpus in data/corpus.json for a fixed
list of hand-written questions, then merges in hand-labeled ground truth
(which context chunk ids are actually relevant to each question, and a
human-written reference answer). For a deliberately chosen handful of
questions it also injects a real factual error or an irrelevant retrieval
set, so the golden set contains genuine negative examples for the metrics
to catch -- not just easy cases.

Run with: python scripts/build_golden_set.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evalharness.pipeline import Corpus, extractive_answer

HERE = os.path.dirname(__file__)
CORPUS_PATH = os.path.join(HERE, "..", "data", "corpus.json")
OUTPUT_PATH = os.path.join(HERE, "..", "data", "golden_set.json")

# Each entry: (question, relevant_chunk_ids, mode)
# mode:
#   "clean"        -> use the real pipeline's extractive answer as-is, full retrieval
#   "fabricated"   -> use the real extractive answer but append one fabricated,
#                     unsupported sentence (a genuine hallucination example)
#   "off_topic"    -> retrieve against a misleading query so retrieved context
#                     does not match relevant_chunk_ids (tests context precision/recall)
QUESTIONS = [
    ("What is the closest planet to the Sun and what are its surface temperatures like?", ["c01"], "clean"),
    ("Why is Venus the hottest planet in the Solar System?", ["c02"], "clean"),
    ("What percentage of Earth's surface is covered by water?", ["c03"], "clean"),
    ("Why does Mars appear red, and what is its tallest volcano?", ["c04"], "clean"),
    ("What is the Great Red Spot on Jupiter?", ["c05", "c22"], "clean"),
    ("What makes Saturn's moon Titan unusual?", ["c06"], "clean"),
    ("Why does Uranus have extreme seasons?", ["c07"], "clean"),
    ("What is notable about Neptune's winds?", ["c08"], "clean"),
    ("What is the largest object in the asteroid belt?", ["c09"], "clean"),
    ("When and why was Pluto reclassified as a dwarf planet?", ["c10"], "clean"),
    ("How does the Sun generate its energy?", ["c11"], "clean"),
    ("Who were the astronauts on the Apollo 11 mission?", ["c12"], "clean"),
    ("What outer planets did Voyager 2 visit?", ["c13"], "clean"),
    ("At what altitude does the Hubble Space Telescope orbit?", ["c14"], "clean"),
    ("How often does the International Space Station orbit the Earth?", ["c15"], "clean"),
    ("What did the Perseverance rover bring to Mars that achieved a first?", ["c16"], "clean"),
    ("How far away is Alpha Centauri in light-years?", ["c17"], "clean"),
    ("How was the first direct image of a black hole obtained?", ["c18"], "clean"),
    ("Why does the James Webb Space Telescope observe in infrared?", ["c19"], "clean"),
    ("How is the Kuiper Belt different from the Oort Cloud?", ["c20"], "clean"),
    ("What triggers a supernova?", ["c21"], "clean"),
    ("Why is Europa considered a candidate for life beyond Earth?", ["c23"], "clean"),
    ("How did the Moon likely form?", ["c24"], "clean"),
    ("How often does Halley's Comet return to the inner Solar System?", ["c25"], "clean"),
    ("How many moons does Jupiter have and what are the four largest called?", ["c05"], "clean"),
    ("What is a light-year and why is it used in astronomy?", ["c17"], "clean"),
    ("What powers the Voyager probes now that they are in interstellar space?", ["c13"], "clean"),
    ("What is the significance of the second Lagrange point for JWST?", ["c19"], "clean"),
    ("Who first discovered Uranus and how?", ["c07"], "clean"),
    ("What is Mercury's day length compared to Earth's?", ["c01"], "clean"),
    # Fabrication / hallucination demo cases: the real extractive answer is
    # still generated from real context, then a single invented sentence is
    # appended that is not supported by any retrieved chunk.
    ("What is the rotation direction of Venus and why is it unusual?", ["c02"], "fabricated"),
    ("What is Mars's largest volcano and how tall is it?", ["c04"], "fabricated"),
    ("When did the Apollo 11 Moon landing happen and who was aboard?", ["c12"], "fabricated"),
    # Off-topic retrieval demo cases: the query used for retrieval is
    # deliberately mismatched to what is actually asked, producing
    # genuinely low context precision/recall against the real labels.
    ("What spacecraft have visited Neptune up close?", ["c08", "c13"], "off_topic"),
    ("What is the composition of Saturn's rings?", ["c06"], "off_topic"),
]

FABRICATED_SENTENCES = {
    "What is the rotation direction of Venus and why is it unusual?": (
        "Venus also has three small natural moons that orbit it in a tight triangular formation."
    ),
    "What is Mars's largest volcano and how tall is it?": (
        "This volcano, Olympus Mons, last erupted in 1972 and is still classified as fully active today."
    ),
    "When did the Apollo 11 Moon landing happen and who was aboard?": (
        "The Apollo 11 crew stayed on the lunar surface for exactly fourteen days before returning to Earth."
    ),
}

OFF_TOPIC_QUERIES = {
    "What spacecraft have visited Neptune up close?": "ocean currents and marine biology of coral reefs",
    "What is the composition of Saturn's rings?": "ancient Roman aqueduct engineering techniques",
}


def main() -> None:
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    corpus = Corpus(ids=[c["id"] for c in chunks], texts=[c["text"] for c in chunks])

    records = []
    for i, (question, relevant_ids, mode) in enumerate(QUESTIONS, start=1):
        retrieval_query = OFF_TOPIC_QUERIES.get(question, question) if mode == "off_topic" else question
        retrieved = corpus.retrieve(retrieval_query, k=3)

        if not retrieved:
            # Off-topic query may not retrieve anything above the sim>0
            # cutoff; fall back to the three least-related chunks to still
            # exercise a real (and genuinely poor) precision/recall case.
            retrieved = [(cid, text, 0.0) for cid, text in zip(corpus.ids[:3], corpus.texts[:3])]

        retrieved_context = [{"id": cid, "text": text} for cid, text, _score in retrieved]
        answer_source_chunks = [c["text"] for c in retrieved_context]

        answer = extractive_answer(question, answer_source_chunks, n_sentences=2)
        if mode == "fabricated":
            answer = f"{answer} {FABRICATED_SENTENCES[question]}"

        records.append(
            {
                "id": f"q{i:02d}",
                "question": question,
                "retrieved_context": retrieved_context,
                "answer": answer,
                "relevant_context_ids": relevant_ids,
                "case_type": mode,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
