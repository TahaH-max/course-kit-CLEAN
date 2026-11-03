"""Answer a question by routing it to SQL or to verified retrieval."""

import re
from datetime import date
from typing import Optional

from .grounding import offline_answer, verify_answer
from .llm import ChatLLM, LLMError
from .router import route_question, strip_course_mentions
from .schemas import AskResponse, CheckResult, DeadlineOut, LLMAnswer, SourceOut
from .store import Store

POLICY_PROMPT = """You help a student understand their course syllabi.
Answer using ONLY the numbered syllabus passages provided.
- If the passages don't answer the question, set "answerable" to false and "answer" to "".
- Never use outside knowledge or general advice about how courses usually work.
- Support every claim with a citation: the passage label (e.g. "S2") and a short quote copied EXACTLY.
- Only state numbers that appear in the passages. Name the course when it matters.
- Keep it to 1-3 sentences."""

NOT_FOUND = "I couldn't find that in your syllabi. Check with the instructor or the course website."
UNVERIFIED = "I found related syllabus text but couldn't verify an answer against it. See the passages below."


def _fmt(d: str) -> str:
    return date.fromisoformat(d).strftime("%a, %b %-d")


def _answer_deadlines(question, route, store: Store) -> AskResponse:
    rows = store.deadlines(
        start=route.start.isoformat() if route.start else None,
        end=route.end.isoformat() if route.end else None,
        courses=route.courses or None,
        kinds=route.kinds or None,
    )
    if route.item:
        word, num = route.item
        word = {"hw": "homework", "assignment": "assignment"}.get(word, word)
        rows = [r for r in rows if word in r["title"].lower() and re.search(rf"\b{num}\b", r["title"])]
    rows = [r for r in rows if route.matches(r["title"])]
    if route.first_only and rows:
        if route.courses:
            seen, firsts = set(), []
            for r in rows:
                if r["course"] not in seen:
                    seen.add(r["course"])
                    firsts.append(r)
            rows = firsts
        else:
            rows = rows[:1]

    deadlines = [DeadlineOut(**r) for r in rows]
    if not deadlines:
        status = "not_found" if route.item else "answered"
        window = f" between {_fmt(route.start.isoformat())} and {_fmt(route.end.isoformat())}" if route.start and route.end else ""
        answer = NOT_FOUND if route.item else f"Nothing matching that is due{window}."
    else:
        status = "answered"
        lines = [f"{_fmt(d.date)}: {d.course} {d.title}" for d in deadlines]
        answer = "\n".join(lines)
    return AskResponse(route="deadlines", status=status, answer=answer, deadlines=deadlines,
                       explanation=route.explain())


def _answer_grading(question, route, store: Store) -> AskResponse:
    rows = store.grading(route.courses or None)
    if route.words:
        # "final report" -> components containing both words; fall back to all if nothing matches
        rows = [r for r in rows if route.matches(r["component"])] or rows
    if not rows:
        return AskResponse(route="grading", status="not_found", answer=NOT_FOUND, explanation=route.explain())

    lines = [f"{r['course']}: {r['component']} is {r['weight']:g}%" for r in rows]
    warnings = [w for c in store.courses() if c["code"] in {r["course"] for r in rows}
                for w in c["warnings"] if "add up to" in w]
    answer = "\n".join(lines + [f"Note ({w})" for w in warnings])
    return AskResponse(route="grading", status="answered", answer=answer, grading=rows,
                       explanation=route.explain())


def _answer_policy(question, route, store: Store, llm: Optional[ChatLLM], k: int) -> AskResponse:
    threshold = store.embedder.min_score
    query = strip_course_mentions(question, route.courses, store.course_index()) if route.courses else question
    hits = store.search(query, k=k, courses=route.courses or None)
    relevant = [h for h in hits if h.score >= threshold]
    sources = {f"S{i + 1}": h for i, h in enumerate(relevant)}
    sources_out = [SourceOut(source_id=s, document=h.document, section=h.section,
                             score=round(h.score, 3), text=h.text) for s, h in sources.items()]
    gate = CheckResult(name="retrieval_gate", passed=bool(relevant),
                       detail=f"best score {hits[0].score if hits else 0:.3f} vs threshold {threshold:.2f}")
    base = dict(route="policy", explanation=route.explain())
    if not relevant:
        return AskResponse(status="not_found", answer=NOT_FOUND, checks=[gate], **base)

    if llm is None:
        result = offline_answer(question, sources)
    else:
        context = "\n\n".join(f"[{s}] ({h.course} / {h.section})\n{h.text}" for s, h in sources.items())
        try:
            result = llm.structured(POLICY_PROMPT, f"Passages:\n\n{context}\n\nQuestion: {question}", LLMAnswer)
        except LLMError as e:
            return AskResponse(status="unverified", answer=UNVERIFIED, sources=sources_out,
                               checks=[gate, CheckResult(name="schema_valid", passed=False, detail=str(e))], **base)

    if not result.answerable:
        return AskResponse(status="not_found", answer=NOT_FOUND, sources=sources_out,
                           checks=[gate, CheckResult(name="model_says_answerable", passed=False,
                                                     detail="passages don't cover this")], **base)
    checks = [gate] + verify_answer(result, sources)
    ok = all(c.passed for c in checks)
    return AskResponse(status="grounded" if ok else "unverified", answer=result.answer if ok else UNVERIFIED,
                       citations=result.citations, sources=sources_out, checks=checks, **base)


def answer_question(question: str, store: Store, llm: Optional[ChatLLM], today: date, k: int = 4) -> AskResponse:
    route = route_question(question, today, store.course_index())
    if route.kind == "deadlines":
        return _answer_deadlines(question, route, store)
    if route.kind == "grading":
        return _answer_grading(question, route, store)
    return _answer_policy(question, route, store, llm, k)
