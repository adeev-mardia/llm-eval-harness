# llm-eval-harness

A small, offline-first evaluation harness for RAG and LLM pipeline outputs,
in the spirit of RAGAS-style evaluation, implemented from scratch with
classic NLP techniques instead of a hosted model call.

It scores (question, retrieved_context, generated_answer) records on:

- Faithfulness - does the answer only state things the retrieved context
  actually supports?
- Answer relevance - does the answer actually address the question that
  was asked?
- Context precision / recall - against a hand-labeled "which chunks are
  actually relevant" ground truth, how good was retrieval?
- Unsupported-claim flagging - the exact sentence(s) in an answer that
  weren't backed by context, so a developer knows precisely what to fix.

Everything above is computed without calling any LLM, using TF-IDF
cosine similarity (scikit-learn) and lexical n-gram overlap. An optional,
real OpenAICompatibleJudge backend lets you swap in a hosted or local
model as the scorer when you have one - same metrics, same interface,
different scoring engine underneath.

## Why this exists

RAG pipelines fail silently. A retriever can pull back the wrong chunks, or
a generator can state something fluent and wrong that sounds exactly like
the rest of the answer. Unit tests don't catch this because there's no
single "correct" string to assert against - you need metrics that look at
how well the output is grounded, not whether it matches a fixed string.
This harness gives you that, cheaply enough to run on every commit, and
transparently enough that you can read exactly why a score came out the
way it did.

## How the metrics actually work

### Faithfulness

1. The answer is split into sentence-level claims using punctuation
   boundaries (evalharness.textutils.split_into_sentences). Sentence-level
   decomposition is the standard choice for this kind of scoring: each
   sentence is treated as one atomic factual statement to check.
2. For each claim, a support score is computed against every retrieved
   context chunk and the best match is kept. The support score blends two
   signals, equally weighted:
   - TF-IDF cosine similarity between the claim and the chunk
     (sklearn.feature_extraction.text.TfidfVectorizer +
     cosine_similarity) - this captures topical/distributional overlap
     and is robust to paraphrasing.
   - Bigram overlap on content words (stopwords removed) - the fraction
     of the claim's word bigrams that also appear in the chunk. This
     catches claims that reuse the context's exact phrasing, which cosine
     similarity over short, sparse TF-IDF vectors can under-score.
3. A claim is supported if its best score clears a threshold (default
   0.28, see evalharness.metrics.DEFAULT_SUPPORT_THRESHOLD). This
   threshold was chosen empirically against the golden set below: it is
   low enough that genuinely supported claims (which reuse several of the
   context's content words) clear it comfortably, and high enough that an
   answer sentence sharing only a stray word or two with the context does
   not.
4. Faithfulness score = (number of supported claims) / (total claims).

Unsupported claims are kept, with their text, in
FaithfulnessResult.unsupported_claims - this is the hallucination report:
you get the literal sentence that wasn't backed by anything retrieved, not
just a number.

### Answer relevance

TF-IDF cosine similarity between the question and the answer, fit locally
on just that pair. This rewards an answer that shares the question's key
nouns/entities and penalizes an answer that is fluent and well-formed but
about something else entirely - a common and otherwise hard-to-catch RAG
failure mode where the retriever pulls near-miss context and the generator
confidently answers the wrong question.

### Context precision / recall

Given a hand-labeled ground truth set of context-chunk ids that are
actually relevant to a question, and the ids of the chunks the retriever
actually returned:

```
precision = |retrieved and relevant| / |retrieved|
recall    = |retrieved and relevant| / |relevant|
```

This is exact set arithmetic, no heuristics - it is only as good as the
ground-truth labels, which is why the golden set below was hand-labeled
chunk by chunk rather than generated.

### Aggregate score

Each record's aggregate is the mean of whichever metrics were actually
computed for it (faithfulness and relevance always; context precision/
recall only when ground-truth labels were supplied for that record). The
report's overall aggregate_score is the mean of all per-record
aggregates - this is the single number the CI gate checks.

## The golden evaluation set

data/golden_set.json contains 35 hand-constructed records built from
a small real pipeline (evalharness/pipeline.py): TF-IDF retrieval over a
25-chunk factual corpus about astronomy and space exploration
(data/corpus.json), with extractive answer generation (the answer is the
context sentence(s) most similar to the question - a legitimate, real
extractive-QA technique, not a template).

Each record is hand-labeled with the chunk ids that are genuinely relevant
to its question. Three categories of record are included on purpose:

- 29 "clean" records - realistic retrieval and extraction, run through
  the pipeline unmodified.
- 3 "fabricated" records - the real extractive answer with one
  additional, deliberately invented sentence appended (e.g. a made-up date
  or a nonexistent moon), to verify that faithfulness scoring actually
  catches a planted hallucination.
- 2 "off_topic" records - retrieval was run against a deliberately
  mismatched query, so the retrieved chunks genuinely do not match the
  hand-labeled relevant set, to verify that context precision/recall and
  answer relevance catch bad retrieval.

Rebuild it (e.g. after editing the corpus or question list) with:

```bash
python scripts/build_golden_set.py
```

### Real baseline scores (from an actual run)

```
$ python -m evalharness.cli --dataset data/golden_set.json --threshold 0.50 --fail-under

Judge backend : heuristic-tfidf
Records scored: 35
Aggregate score         : 0.5954
Mean faithfulness        : 0.9714
Mean answer relevance    : 0.2872
Mean context precision   : 0.3857
Mean context recall      : 0.9429
```

A few things worth noting about these real numbers:

- Faithfulness is high (0.9714) because most records are "clean"
  extractive answers that literally reuse context sentences - exactly what
  faithfulness is supposed to reward.
- Answer relevance is lower (0.2872) because TF-IDF cosine similarity
  between a short question and a short answer, after stopwords are
  removed, is a fairly strict metric - it rewards shared content words, not
  topical "vibes." The two deliberately off-topic records score 0.0 on
  relevance, which is exactly the point: they show the metric does its job.
- Context precision (0.3857) is noticeably lower than recall (0.9429)
  because the pipeline retrieves k=3 chunks for every question, but most
  questions only have one or two genuinely relevant chunks - so some
  retrieved padding is expected and precision reflects that honestly,
  while recall shows the one relevant chunk is almost always in that
  top-3.
- The worst-scoring records in every run are, as intended, the 3
  fabricated-claim records and the 2 off-topic-retrieval records - run
  python -m evalharness.cli --dataset data/golden_set.json yourself and
  check the "Worst N records" section; it lists their exact unsupported
  sentences.

## Using it on your own pipeline's outputs

```python
from evalharness import EvalRecord, evaluate_dataset
from evalharness.report import ContextChunk

records = [
    EvalRecord(
        id="r1",
        question="What is the boiling point of water at sea level?",
        retrieved_context=[
            ContextChunk(id="chunk-3", text="Water boils at 100C (212F) at standard atmospheric pressure."),
        ],
        answer="Water boils at 100 degrees Celsius at sea level.",
        relevant_context_ids=["chunk-3"],  # optional, enables context precision/recall
    ),
]

report = evaluate_dataset(records)
print(report.aggregate_score, report.mean_faithfulness)
for claim in report.record_scores[0].faithfulness.unsupported_claims:
    print("UNSUPPORTED:", claim.text)
```

Or from the command line against a JSON file shaped like
data/golden_set.json:

```bash
python -m evalharness.cli --dataset path/to/your_dataset.json --worst 5
python -m evalharness.cli --dataset path/to/your_dataset.json --json > report.json
```

## Plugging in a real LLM judge

By default, evaluate_dataset(records) uses HeuristicJudge - fully
offline, no API key, no network call. If you have access to a hosted
model (OpenAI, Azure OpenAI) or a local OpenAI-compatible server (Ollama,
vLLM, llama.cpp, LM Studio), you can swap in
OpenAICompatibleJudge instead, which scores the same two metrics by
asking the model directly over the standard Chat Completions HTTP API:

```python
from evalharness import evaluate_dataset
from evalharness.judge import OpenAICompatibleJudge

# Hosted OpenAI
judge = OpenAICompatibleJudge(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

# Local Ollama server - no api_key needed
judge = OpenAICompatibleJudge(
    model="llama3.1",
    base_url="http://localhost:11434/v1",
)

report = evaluate_dataset(records, judge=judge)
```

This repo's own test suite (tests/test_judge.py) verifies the judge's
request construction and response parsing against a stubbed requests.post
- the usual way to unit test an HTTP client without a live network call -
so the code path is exercised even though no hosted endpoint was reachable
while building this project. Context precision/recall doesn't need a judge
at all, since it's exact set arithmetic against your own ground-truth
labels.

## CI-gated evaluation

.github/workflows/eval-gate.yml runs on every push and pull request:

1. Installs the package and runs the full pytest suite.
2. Runs python -m evalharness.cli --dataset data/golden_set.json --threshold 0.50 --fail-under
   against the golden set.
3. Fails the build if the aggregate score drops below 0.50 - a real,
   working quality gate, not a placeholder step. A pull request that
   regresses the metrics' logic, or edits the golden set in a way that
   tanks the score, fails CI the same way a broken unit test would.

The threshold is intentionally set below the current baseline (0.5954)
rather than right at it, so normal floating-point/library-version noise
doesn't cause spurious failures, while a real regression (e.g. breaking the
support-threshold logic, or corrupting several golden-set answers) still
trips it.

## Running the tests yourself

```bash
pip install -e ".[dev]"
pytest -v
```

40 tests cover: claim splitting, faithfulness on fully-supported /
fabricated / context-free / empty-answer inputs, answer relevance on
on-topic vs. off-topic answers, context precision/recall edge cases
(perfect match, partial match, no overlap, deduped ids), the report
aggregation logic, the retrieval/extraction pipeline, the LLM-judge
backend's request/response handling (via a stubbed transport), and the
golden set itself (well-formedness, and that fabricated/off-topic cases
really do score worse than clean ones).

## Limitations

- TF-IDF is a bag-of-words method. It has no notion of negation, so
  "Mercury has an atmosphere" and "Mercury has no atmosphere" can register
  as similar. A production system handling claims where negation matters
  should prefer the LLM-judge backend or add an explicit negation check.
- Sentence-level claim splitting is heuristic, not a trained sentence
  tokenizer - it will occasionally mis-split unusual punctuation (e.g.
  abbreviations like "Dr." or decimal numbers in rare phrasings).
- The support threshold (0.28) was tuned on one golden set in one
  domain (factual astronomy Q&A). A different domain or answer style
  (e.g. much longer claims, or heavily paraphrased answers) may need a
  different threshold - it's a single documented constant, not a buried
  magic number, specifically so it's easy to re-tune.
- Context precision/recall is only as good as the ground-truth labels.
  It cannot tell you a chunk is relevant if nobody labeled it that way.
- The LLM-judge backend is implemented but untested against a live
  endpoint in this repository's own development, because the build
  environment used to create it has no outbound access to hosted LLM
  APIs. Its request construction and response parsing are verified with a
  stubbed HTTP transport instead; if you point it at a real OpenAI, Azure
  OpenAI, or local server, double-check your first few scores by eye.

## Project layout

```
evalharness/
  metrics.py     faithfulness, answer relevance, context precision/recall
  textutils.py   sentence splitting, tokenization, n-gram overlap
  judge.py       HeuristicJudge (offline) and OpenAICompatibleJudge (LLM)
  pipeline.py    the small TF-IDF retrieval + extractive-answer pipeline
                 used only to build the golden set
  report.py      EvalRecord / EvalReport / evaluate_dataset
  cli.py         command-line entry point, including the CI gate flags
data/
  corpus.json      25-chunk factual corpus (astronomy/space exploration)
  golden_set.json  35 hand-labeled eval records built from that corpus
scripts/
  build_golden_set.py   regenerates data/golden_set.json
tests/
  test_metrics.py, test_judge.py, test_report.py, test_pipeline.py,
  test_golden_set.py
.github/workflows/eval-gate.yml   the CI quality gate
```

## License

MIT, see LICENSE.
