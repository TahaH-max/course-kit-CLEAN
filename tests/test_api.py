"""HTTP-level tests (offline mode, no API key needed)."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COURSEKIT_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("LLM_PROVIDER", "offline")
    monkeypatch.setenv("COURSEKIT_TODAY", "2026-11-03")
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_seeded_courses(client):
    assert client.get("/health").json()["today"] == "2026-11-03"
    courses = client.get("/courses").json()
    assert [c["code"] for c in courses] == ["CSC 216", "ENG 205", "MAT 230"]
    eng = next(c for c in courses if c["code"] == "ENG 205")
    assert any("95%" in w for w in eng["warnings"])


def test_deadlines_filters(client):
    rows = client.get("/deadlines", params={"start": "2026-11-09", "end": "2026-11-15"}).json()
    assert len(rows) == 4
    exams = client.get("/deadlines", params={"kind": "exam", "course": "MAT 230"}).json()
    assert [r["title"] for r in exams][:2] == ["Midterm 1", "Midterm 2"]


def test_calendar_download(client):
    r = client.get("/calendar.ics", params={"course": "CSC 216"})
    assert r.headers["content-type"].startswith("text/calendar")
    assert r.text.count("BEGIN:VEVENT") == 9


def test_ask_routes(client):
    assert client.post("/ask", json={"question": "What's due next week?"}).json()["route"] == "deadlines"
    assert client.post("/ask", json={"question": "How much is the final exam worth in CSC 216?"}).json()["route"] == "grading"
    policy = client.post("/ask", json={"question": "Can I use a calculator on probability exams?"}).json()
    assert policy["route"] == "policy" and policy["status"] == "grounded"


def test_upload_new_syllabus_then_delete(client):
    syllabus = (b"# BIO 101\n\nCourse: BIO 101 - General Biology\nInstructor: Dr. Ana Ruiz\n"
                b"Term: Fall 2026 (Aug 24 - Dec 18, 2026)\n\n## Grading\n\n- Labs: 40%\n- Final exam: 60%\n\n"
                b"## Lab Safety\n\nClosed-toe shoes and goggles are required in every lab.\n\n"
                b"## Schedule\n\n- Mon Nov 9: Lab report 3 due\n- Fri Dec 11: Final Exam\n")
    r = client.post("/courses", files={"file": ("bio101.md", syllabus, "text/markdown")})
    assert r.status_code == 201
    course = r.json()
    assert course["code"] == "BIO 101" and course["deadline_count"] == 2 and course["warnings"] == []
    ans = client.post("/ask", json={"question": "What shoes do I need for biology lab?"}).json()
    assert ans["status"] == "grounded" and "goggles" in ans["answer"]
    assert client.delete(f"/courses/{course['id']}").status_code == 204
    assert client.delete(f"/courses/{course['id']}").status_code == 404


def test_upload_rejects_unsupported_type(client):
    assert client.post("/courses", files={"file": ("x.docx", b"data", "application/octet-stream")}).status_code == 400


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
