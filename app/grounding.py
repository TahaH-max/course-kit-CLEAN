"""Verification helpers for LLM answers over retrieved passages.

Shared idea with the extraction checks in syllabus.py: nothing the model says
is shown unless it can be traced back to the source text.
- every citation must point to a passage that was actually retrieved
- every quote must appear in that passage
- every number in the answer must appear in a cited passage
"""

import re
from difflib import SequenceMatcher

from .embeddings import tokenize
from .schemas import CheckResult, Citation, LLMAnswer
from .store import Hit


# ---------- text helpers ----------

def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[^a-z0-9$%.']+", " ", text)
    text = re.sub(r"\.(?=\s|$)", " ", text)  # drop sentence-ending periods, keep decimals
    return re.sub(r"\s+", " ", text).strip()


def _numbers(text: str) -> set[str]:
    text = re.sub(r"\[S\d+\]", " ", text)  # ignore inline citation labels
    found = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
    out = set()
    for n in found:
        n = n.replace(",", "")
        if "." in n:
            n = n.rstrip("0").rstrip(".")
        out.add(n)
    return out


def _quote_in_passage(quote: str, passage: str) -> bool:
    q, p = _normalize(quote), _normalize(passage)
    if not q:
        return False
    if q in p:
        return True
    # tolerate tiny differences (punctuation, a dropped word) but not paraphrase
    m = SequenceMatcher(None, q, p).find_longest_match(0, len(q), 0, len(p))
    return m.size / len(q) >= 0.9


def _split_sentences(text: str) -> list[str]:
    # split on line breaks and sentence ends, but not after titles like "Dr." or "Prof."
    parts = re.split(r"\n+|(?<!\bDr\.)(?<!\bProf\.)(?<!\bMr\.)(?<!\bMs\.)(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s and s.strip()]


# ---------- verification ----------

def verify_answer(result: LLMAnswer, sources: dict[str, Hit]) -> list[CheckResult]:
    checks: list[CheckResult] = []

    checks.append(
        CheckResult(
            name="has_citations",
            passed=bool(result.citations),
            detail=f"{len(result.citations)} citation(s)",
        )
    )

    unknown = [c.source_id for c in result.citations if c.source_id not in sources]
    checks.append(
        CheckResult(
            name="citations_point_to_retrieved_sources",
            passed=not unknown,
            detail=f"unknown source ids: {unknown}" if unknown else "all ids valid",
        )
    )

    bad_quotes = [
        c.quote
        for c in result.citations
        if c.source_id in sources and not _quote_in_passage(c.quote, sources[c.source_id].text)
    ]
    checks.append(
        CheckResult(
            name="quotes_found_in_sources",
            passed=not bad_quotes and not unknown,
            detail=f"not found: {bad_quotes}" if bad_quotes else "all quotes matched",
        )
    )

    cited_text = " ".join(sources[c.source_id].text for c in result.citations if c.source_id in sources)
    missing = sorted(_numbers(result.answer) - _numbers(cited_text))
    checks.append(
        CheckResult(
            name="numbers_supported_by_citations",
            passed=not missing,
            detail=f"unsupported numbers: {missing}" if missing else "all numbers supported",
        )
    )
    return checks


# ---------- offline fallback ----------

def offline_answer(question: str, sources: dict[str, Hit]) -> LLMAnswer:
    """Extractive fallback when no LLM is configured: return the best-matching sentence(s).

    It can only quote the documents, so it is grounded by construction, but it
    can't synthesize across passages the way the model can.
    """
    q_tokens = set(tokenize(question))
    scored = []
    for sid, hit in sources.items():
        body = hit.text.split("\n", 1)[1] if hit.section and "\n" in hit.text else hit.text
        for sent in _split_sentences(body):
            overlap = len(q_tokens & set(tokenize(sent)))
            if overlap:
                scored.append((overlap + hit.score, sid, sent))
    if not scored:
        return LLMAnswer(answerable=False, answer="", citations=[])
    scored.sort(key=lambda x: -x[0])
    best = scored[0]
    picked = [best]
    # include a second sentence from the same passage if it is nearly as relevant
    for cand in scored[1:]:
        if cand[1] == best[1] and cand[0] >= 0.75 * best[0]:
            picked.append(cand)
            break
    return LLMAnswer(
        answerable=True,
        answer=" ".join(s for _, _, s in picked),
        citations=[Citation(source_id=sid, quote=s) for _, sid, s in picked],
    )


