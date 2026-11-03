"""Evaluate routing and answers on a fixed question set.

Usage:  python -m evals.run_eval
        LLM_PROVIDER=offline python -m evals.run_eval

Scored separately, because they fail for different reasons:
- routing:     did the question go to the right path (deadlines / grading / policy)?
- deadlines:   exact match on the set of returned (course, date) pairs
- grading/policy: answer contains the expected fact and status is answered/grounded
- unanswerable: nothing shown (not_found, or an empty deadline list)
"""

import json
import sys
from datetime import date
from pathlib import Path

from app.answer import answer_question
from app.config import get_settings
from app.embeddings import get_embedder
from app.ingest import chunk_text
from app.llm import get_llm
from app.store import Store
from app.syllabus import extract_syllabus

HERE = Path(__file__).parent


def main() -> int:
    settings = get_settings()
    llm = get_llm(settings)
    store = Store(":memory:", get_embedder(settings))
    for path in sorted(settings.sample_dir.glob("*.md")):
        text = path.read_text()
        extract, warnings = extract_syllabus(text, llm)
        store.add_syllabus(path.name, extract, warnings, chunk_text(text))

    spec = json.loads((HERE / "questions.json").read_text())
    today = date.fromisoformat(spec["today"])
    cases = spec["cases"]
    routed = correct = 0
    answerable = [c for c in cases if not c.get("unanswerable")]
    unanswerable = [c for c in cases if c.get("unanswerable")]
    refused = 0
    rows = []

    for c in cases:
        resp = answer_question(c["q"], store, llm, today, k=settings.top_k)
        route_ok = resp.route == c["route"]
        routed += route_ok
        if c.get("unanswerable"):
            ok = resp.status == "not_found" or (resp.route == "deadlines" and not resp.deadlines)
            refused += ok
        elif "expect_items" in c:
            # Match on course + date, not title text: LLM extraction may word titles differently.
            ok = sorted(f"{d.course} {d.date}" for d in resp.deadlines) == sorted(c["expect_items"])
            correct += ok
        else:
            # "expect" is one phrase or a list of acceptable phrases (LLM answers paraphrase).
            phrases = c["expect"] if isinstance(c["expect"], list) else [c["expect"]]
            ok = resp.status in ("answered", "grounded") and any(p.lower() in resp.answer.lower() for p in phrases)
            correct += ok
        mark = "ok  " if ok and route_ok else "MISS"
        print(f"{mark} {resp.route:9} {resp.status:10} {c['q']}")
        rows.append({"q": c["q"], "route": resp.route, "expected_route": c["route"], "status": resp.status,
                     "ok": ok, "answer": resp.answer, "explanation": resp.explanation})

    summary = {
        "provider": llm.provider if llm else "offline",
        "embedder": store.embedder.name,
        "routing_accuracy": f"{routed}/{len(cases)}",
        "correct_answers": f"{correct}/{len(answerable)}",
        "correct_refusals": f"{refused}/{len(unanswerable)}",
    }
    print("\n" + json.dumps(summary, indent=2))
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())