from datetime import date

from app.router import route_question

from .conftest import TODAY

COURSES = [{"code": "CSC 216", "title": "Data Structures"},
           {"code": "MAT 230", "title": "Introduction to Probability"},
           {"code": "ENG 205", "title": "Technical Writing"}]


def r(q):
    return route_question(q, TODAY, COURSES)


def test_next_week_is_monday_to_sunday():
    route = r("What's due next week?")
    assert route.kind == "deadlines"
    assert (route.start, route.end) == (date(2026, 11, 9), date(2026, 11, 15))


def test_this_week_runs_from_today_to_sunday():
    assert (r("What do I have this week?").start, r("What do I have this week?").end) == (TODAY, date(2026, 11, 8))


def test_month_range():
    route = r("What exams are in December?")
    assert (route.start, route.end, route.kinds) == (date(2026, 12, 1), date(2026, 12, 31), ["exam"])


def test_named_item_searches_all_dates():
    route = r("When is midterm 2?")
    assert route.item == ("midterm", "2") and route.start is None


def test_next_quiz_returns_first_only():
    route = r("When is my next quiz?")
    assert route.first_only and route.kinds == ["quiz"]


def test_exam_policy_question_is_not_a_deadline_question():
    assert r("Can I bring notes to the final exam?").kind == "policy"
    assert r("Can I use a calculator on exams?").kind == "policy"


def test_grading_route():
    route = r("How much is the final exam worth?")
    assert route.kind == "grading" and route.words == ["final"]


def test_course_detection_by_code_number_and_title():
    assert r("late policy for CSC 216").courses == ["CSC 216"]
    assert r("late policy for csc216").courses == ["CSC 216"]
    assert r("when is the 230 final").courses == ["MAT 230"]
    assert r("calculator rules in probability").courses == ["MAT 230"]
    assert r("what's the attendance policy").courses == []


def test_when_question_without_a_graded_item_searches_text():
    assert r("When is spring break?").kind == "policy"
    assert r("When are office hours for CSC 216?").kind == "policy"


def test_course_mentions_are_stripped_from_the_search_query():
    from app.router import strip_course_mentions
    assert strip_course_mentions("Is there a textbook for CSC 216?", ["CSC 216"], COURSES) == "Is there a textbook for ?"
    assert strip_course_mentions("calculator rules in probability", ["MAT 230"], COURSES) == "calculator rules in"


def test_final_report_excludes_the_draft():
    route = r("When is the ENG 205 final report due?")
    assert route.matches("Final report due by 11:59 PM") and not route.matches("Final report draft due")
