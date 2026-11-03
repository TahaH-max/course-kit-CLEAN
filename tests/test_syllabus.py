from datetime import date

from app.syllabus import classify, extract_syllabus, find_dates

from .conftest import sample_text


def test_bullet_format_syllabus():
    ex, warnings = extract_syllabus(sample_text("csc216_data_structures.md"), None)
    assert ex.course_code == "CSC 216"
    assert ex.instructor == "Dr. Elena Park"
    assert (ex.term_start, ex.term_end) == ("2026-08-24", "2026-12-18")
    assert len(ex.deadlines) == 9
    assert sum(g.weight for g in ex.grading) == 100
    assert warnings == []


def test_table_format_syllabus():
    ex, warnings = extract_syllabus(sample_text("mat230_probability.md"), None)
    assert ex.course_code == "MAT 230"
    assert [g.component for g in ex.grading] == ["Homework", "Quizzes", "Midterm 1", "Midterm 2", "Final exam"]
    midterm2 = next(d for d in ex.deadlines if d.title == "Midterm 2")
    assert midterm2.date == "2026-11-13" and midterm2.kind == "exam"
    assert warnings == []


def test_grading_that_doesnt_sum_to_100_is_flagged():
    _, warnings = extract_syllabus(sample_text("eng205_technical_writing.md"), None)
    assert any("add up to 95%" in w for w in warnings)


def test_lecture_topics_are_not_deadlines():
    text = sample_text("csc216_data_structures.md").replace(
        "- Tue Nov 10: Quiz 8", "- Tue Nov 10: Quiz 8\n- Thu Nov 12: Lecture on heaps")
    ex, _ = extract_syllabus(text, None)
    assert not any("heaps" in d.title for d in ex.deadlines)


def test_classify():
    assert classify("Final report draft due") == "assignment"
    assert classify("Final Exam, 8:00 AM") == "exam"
    assert classify("Midterm 2") == "exam"
    assert classify("Midterm project due") == "assignment"
    assert classify("Quiz 8") == "quiz"
    assert classify("Lecture: Heaps") == "other"


def test_find_dates_formats():
    assert find_dates("Thu Nov 5: PA5 due", 2026) == [date(2026, 11, 5)]
    assert find_dates("November 5th, 2027", 2026) == [date(2027, 11, 5)]
    assert find_dates("due 2026-11-05", 2026) == [date(2026, 11, 5)]
    assert find_dates("due 11/5", 2026) == [date(2026, 11, 5)]
    assert find_dates("Feb 30", 2026) == []  # impossible date is ignored, not crashed on


# ---------- LLM extraction is verified against the document ----------

def _llm_extract(**overrides):
    base = {
        "course_code": "MAT 230", "title": "Introduction to Probability", "instructor": "Prof. Samuel Okafor",
        "term_start": "2026-08-24", "term_end": "2026-12-18",
        "grading": [{"component": "Homework", "weight": 20}, {"component": "Quizzes", "weight": 10},
                    {"component": "Midterm 1", "weight": 20}, {"component": "Midterm 2", "weight": 20},
                    {"component": "Final exam", "weight": 30}],
        "deadlines": [{"title": "Midterm 2", "date": "2026-11-13", "kind": "exam", "source_line": "| Nov 13 | Midterm 2 |"}],
    }
    base.update(overrides)
    return base


def test_llm_extraction_kept_when_it_matches_the_document(fake_llm):
    llm, _ = fake_llm(_llm_extract())
    ex, warnings = extract_syllabus(sample_text("mat230_probability.md"), llm)
    assert [d.title for d in ex.deadlines] == ["Midterm 2"]
    assert warnings == []


def test_llm_deadline_with_invented_source_line_is_dropped(fake_llm):
    llm, _ = fake_llm(_llm_extract(deadlines=[
        {"title": "Project 1", "date": "2026-11-20", "kind": "assignment", "source_line": "Nov 20: Project 1 due"}]))
    ex, warnings = extract_syllabus(sample_text("mat230_probability.md"), llm)
    assert ex.deadlines == []
    assert any("source line isn't in the syllabus" in w for w in warnings)


def test_llm_deadline_with_wrong_date_is_dropped(fake_llm):
    # Real line, but the model put it on the 12th instead of the 13th.
    llm, _ = fake_llm(_llm_extract(deadlines=[
        {"title": "Midterm 2", "date": "2026-11-12", "kind": "exam", "source_line": "| Nov 13 | Midterm 2 |"}]))
    ex, warnings = extract_syllabus(sample_text("mat230_probability.md"), llm)
    assert ex.deadlines == []
    assert any("isn't the date written on its line" in w for w in warnings)


def test_llm_grading_weight_not_in_document_is_flagged(fake_llm):
    grading = [{"component": "Homework", "weight": 25}, {"component": "Quizzes", "weight": 5},
               {"component": "Midterm 1", "weight": 20}, {"component": "Midterm 2", "weight": 20},
               {"component": "Final exam", "weight": 30}]
    llm, _ = fake_llm(_llm_extract(grading=grading))
    _, warnings = extract_syllabus(sample_text("mat230_probability.md"), llm)
    assert any("25%" in w for w in warnings) and any("5%" in w for w in warnings)
