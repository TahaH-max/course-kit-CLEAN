from app.answer import answer_question
from app.calendar_export import to_ics

from .conftest import TODAY


def ask(store, q, llm=None):
    return answer_question(q, store, llm, TODAY)


# ---------- structured routes ----------

def test_due_next_week_across_courses_in_date_order(store):
    resp = ask(store, "What's due next week?")
    assert resp.route == "deadlines" and resp.status == "answered"
    assert [(d.date, d.course) for d in resp.deadlines] == [
        ("2026-11-10", "CSC 216"), ("2026-11-11", "ENG 205"), ("2026-11-11", "MAT 230"), ("2026-11-13", "MAT 230")]
    assert all(d.source_line for d in resp.deadlines)  # every result traces back to a syllabus line


def test_final_does_not_return_midterm(store):
    resp = ask(store, "When is the final for data structures?")
    assert [d.title for d in resp.deadlines] == ["Final Exam, 8:00 AM - 10:30 AM"]


def test_unknown_item_is_not_found(store):
    assert ask(store, "When is Quiz 12?").status == "not_found"


def test_empty_window_is_an_answer_not_an_error(store):
    resp = ask(store, "What's due tomorrow?")  # Wed Nov 4: MAT 230 HW 8
    assert resp.status == "answered" and len(resp.deadlines) == 1
    resp = ask(store, "What's due today?")
    assert resp.status == "answered" and resp.deadlines == [] and "Nothing" in resp.answer


def test_grading_question(store):
    resp = ask(store, "How much is the final exam worth in CSC 216?")
    assert resp.route == "grading"
    assert resp.grading == [{"course": "CSC 216", "component": "Final exam", "weight": 30.0}]


def test_grading_answer_carries_the_95_percent_warning(store):
    assert "95%" in ask(store, "What is the grade breakdown for ENG 205?").answer


# ---------- policy route (retrieval + verification) ----------

LATE_Q = "What is the late policy for programming assignments?"
LATE_QUOTE = "Programming assignments lose 10% per day late, up to 3 days."


def test_policy_offline_is_grounded(store):
    resp = ask(store, LATE_Q)
    assert resp.route == "policy" and resp.status == "grounded"
    assert "10% per day" in resp.answer


def test_off_topic_never_calls_llm(store, fake_llm):
    llm, client = fake_llm()
    resp = ask(store, "What's the parking policy?", llm)
    assert resp.status == "not_found" and client.calls == []


def test_llm_answer_with_wrong_number_is_withheld(store, fake_llm):
    llm, _ = fake_llm({"answerable": True, "answer": "You lose 15% per day, up to 3 days.",
                       "citations": [{"source_id": "S1", "quote": LATE_QUOTE}]})
    resp = ask(store, LATE_Q, llm)
    assert resp.status == "unverified" and "15%" not in resp.answer


def test_llm_answer_with_invented_quote_is_withheld(store, fake_llm):
    llm, _ = fake_llm({"answerable": True, "answer": "You lose 10% per day.",
                       "citations": [{"source_id": "S1", "quote": "Late work is penalized 10% daily."}]})
    assert ask(store, LATE_Q, llm).status == "unverified"


def test_valid_llm_answer_is_grounded_and_prompt_is_course_scoped(store, fake_llm):
    llm, client = fake_llm({"answerable": True, "answer": "Programming assignments lose 10% per day late, up to 3 days.",
                            "citations": [{"source_id": "S1", "quote": LATE_QUOTE}]})
    resp = ask(store, "What's the late policy in CSC 216?", llm)
    assert resp.status == "grounded"
    prompt = client.calls[0]["messages"][1]["content"]
    assert "CSC 216" in prompt and "MAT 230" not in prompt  # retrieval filtered to the named course


# ---------- calendar ----------

def test_ics_export(store):
    ics = to_ics(store.deadlines())
    assert ics.startswith("BEGIN:VCALENDAR\r\n") and ics.endswith("END:VCALENDAR\r\n")
    assert ics.count("BEGIN:VEVENT") == 24
    assert "DTSTART;VALUE=DATE:20261215" in ics
    assert "Final Exam\\, 8:00 AM" in ics  # commas escaped per RFC 5545
    assert to_ics(store.deadlines()).count("UID:") == len(set(l for l in ics.splitlines() if l.startswith("UID:")))


def test_course_name_alone_does_not_pass_the_retrieval_gate(store):
    # Before stripping course names from the query, every CSC 216 chunk matched "CSC 216".
    assert ask(store, "Is there a textbook for CSC 216?").status == "not_found"


def test_instructor_title_does_not_break_sentences(store):
    assert "Dr. Park holds office hours" in ask(store, "What are Dr. Park's office hours?").answer
