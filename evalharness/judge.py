"""Scoring backends.

``HeuristicJudge`` is the default, fully offline backend: it wraps the
TF-IDF / n-gram metrics in :mod:`evalharness.metrics`, so it needs no
network access and no API key.

``OpenAICompatibleJudge`` is a real, pluggable alternative for anyone who
*does* have access to a hosted or local model: it speaks the OpenAI
Chat Completions HTTP API (``POST {base_url}/chat/completions``), which is
also what Azure OpenAI and local servers such as Ollama (at
``http://localhost:11434/v1``) and vLLM/llama.cpp OpenAI-compat servers
implement. It builds the same two metrics (faithfulness, answer relevance)
by asking the model to judge them directly, instead of via TF-IDF.

This file cannot be exercised against a live endpoint inside this build
environment (no outbound access to hosted LLM APIs here), but its request
construction and response parsing are covered for real by
``tests/test_judge.py``, which stubs ``requests.post`` the way the rest of
this project's transport-level tests do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Protocol, Sequence

from .metrics import (
    FaithfulnessResult,
    RelevanceResult,
    answer_relevance_score,
    faithfulness_score,
)


class Judge(Protocol):
    """Interface a scoring backend must satisfy to be usable by the report
    builder in :mod:`evalharness.report`.
    """

    name: str

    def score_faithfulness(self, answer: str, context_chunks: Sequence[str]) -> FaithfulnessResult: ...

    def score_relevance(self, question: str, answer: str) -> RelevanceResult: ...


@dataclass
class HeuristicJudge:
    """Default offline judge: TF-IDF cosine similarity + n-gram overlap.

    This is the backend used everywhere in this project's own golden-set
    evaluation and CI gate, so every baseline score in the README comes
    from this class, not from an LLM call.
    """

    name: str = "heuristic-tfidf"
    support_threshold: float = 0.28

    def score_faithfulness(self, answer: str, context_chunks: Sequence[str]) -> FaithfulnessResult:
        return faithfulness_score(answer, context_chunks, threshold=self.support_threshold)

    def score_relevance(self, question: str, answer: str) -> RelevanceResult:
        return answer_relevance_score(question, answer)


class JudgeTransportError(RuntimeError):
    """Raised when the LLM judge's HTTP call fails or returns an
    unparseable response."""


_FAITHFULNESS_SYSTEM_PROMPT = (
    "You are a strict fact-checking assistant. You will be given a numbered "
    "list of CONTEXT passages and a single CLAIM. Respond with ONLY a JSON "
    "object of the form {\"supported\": true|false, \"best_context_index\": <int or null>} "
    "indicating whether the claim is fully supported by at least one context "
    "passage, and if so which passage (0-indexed) supports it best."
)

_RELEVANCE_SYSTEM_PROMPT = (
    "You are a strict evaluation assistant. You will be given a QUESTION and "
    "an ANSWER. Respond with ONLY a JSON object of the form "
    "{\"relevance\": <float between 0.0 and 1.0>} scoring how directly the "
    "answer addresses the question (1.0 = fully and directly addresses it, "
    "0.0 = does not address it at all)."
)


@dataclass
class OpenAICompatibleJudge:
    """LLM-as-judge backend speaking the OpenAI Chat Completions HTTP API.

    Works against:
      * OpenAI (``base_url="https://api.openai.com/v1"``)
      * Azure OpenAI (``base_url="https://<resource>.openai.azure.com/openai/deployments/<deployment>"``,
        with ``api_key`` set to the Azure key and ``extra_headers`` or
        ``api_key_header`` adjusted as needed)
      * A local server such as Ollama (``base_url="http://localhost:11434/v1"``)
        or any other OpenAI-compatible server (vLLM, llama.cpp, LM Studio)

    Only ``requests`` is used for the HTTP call, imported lazily so that
    environments without network access (and without ``requests``
    installed) can still import and use :class:`HeuristicJudge` without
    error.
    """

    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    timeout: float = 30.0
    name: str = "llm-judge"

    def __post_init__(self) -> None:
        if not self.name or self.name == "llm-judge":
            self.name = f"llm-judge:{self.model}"

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        import requests  # lazy import: optional dependency for offline users

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)
        if response.status_code != 200:
            raise JudgeTransportError(
                f"LLM judge request failed: HTTP {response.status_code} - {response.text[:500]}"
            )
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise JudgeTransportError(f"Unexpected LLM judge response shape: {data!r}") from exc
        return content

    @staticmethod
    def _extract_json(content: str) -> dict:
        """Parse the model's reply as JSON, tolerating a fenced code block
        (some local models wrap JSON in ```json ... ``` even when told not
        to).
        """
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise JudgeTransportError(f"LLM judge did not return valid JSON: {content!r}") from exc

    def score_faithfulness(self, answer: str, context_chunks: Sequence[str]) -> FaithfulnessResult:
        from .metrics import Claim, split_into_claims

        claims_text = split_into_claims(answer)
        if not claims_text:
            return FaithfulnessResult(score=1.0, claims=[])
        if not context_chunks:
            claims = [
                Claim(text=c, support_score=0.0, best_context_index=None, supported=False)
                for c in claims_text
            ]
            return FaithfulnessResult(score=0.0, claims=claims)

        context_block = "\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks))
        claims: List[Claim] = []
        for claim_text in claims_text:
            user_prompt = f"CONTEXT:\n{context_block}\n\nCLAIM:\n{claim_text}"
            content = self._chat(_FAITHFULNESS_SYSTEM_PROMPT, user_prompt)
            parsed = self._extract_json(content)
            supported = bool(parsed.get("supported", False))
            best_idx = parsed.get("best_context_index")
            claims.append(
                Claim(
                    text=claim_text,
                    support_score=1.0 if supported else 0.0,
                    best_context_index=best_idx if isinstance(best_idx, int) else None,
                    supported=supported,
                )
            )
        score = sum(1 for c in claims if c.supported) / len(claims)
        return FaithfulnessResult(score=round(score, 4), claims=claims)

    def score_relevance(self, question: str, answer: str) -> RelevanceResult:
        user_prompt = f"QUESTION:\n{question}\n\nANSWER:\n{answer}"
        content = self._chat(_RELEVANCE_SYSTEM_PROMPT, user_prompt)
        parsed = self._extract_json(content)
        relevance = float(parsed.get("relevance", 0.0))
        relevance = max(0.0, min(1.0, relevance))
        return RelevanceResult(score=round(relevance, 4), detail=f"judged by {self.model}")
