"""FastAPI app for CourseKit.

Run:  uvicorn app.main:app --reload     then open http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs
"""

import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .answer import answer_question
from .calendar_export import to_ics
from .config import get_settings
from .embeddings import get_embedder
from .ingest import chunk_text, extract_text
from .llm import LLMError, get_llm
from .schemas import AskRequest, AskResponse, CourseOut, DeadlineOut
from .store import Store
from .syllabus import extract_syllabus

logging.basicConfig(level=logging.INFO)
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def ingest_syllabus(store: Store, llm, name: str, text: str) -> int:
    extract, warnings = extract_syllabus(text, llm)
    return store.add_syllabus(name, extract, warnings, chunk_text(text))


def today() -> date:
    override = app.state.settings.today_override
    return date.fromisoformat(override) if override else date.today()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.store = Store(settings.db_path, get_embedder(settings))
    app.state.llm = get_llm(settings)
    if not app.state.store.courses():
        for path in sorted(settings.sample_dir.glob("*.md")):
            ingest_syllabus(app.state.store, app.state.llm, path.name, path.read_text())
    logging.info("Provider: %s | embedder: %s", settings.resolved_provider(), app.state.store.embedder.name)
    yield


app = FastAPI(title="CourseKit", version="1.0", lifespan=lifespan,
              description="Syllabus assistant: deadlines and grading from SQL, policies from verified retrieval.")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "provider": app.state.settings.resolved_provider(),
            "embedder": app.state.store.embedder.name, "today": today().isoformat()}


@app.get("/courses", response_model=list[CourseOut])
def list_courses():
    return app.state.store.courses()


@app.post("/courses", response_model=CourseOut, status_code=201)
async def upload_syllabus(file: UploadFile = File(...)):
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (5 MB max).")
    try:
        text = extract_text(file.filename or "syllabus.txt", data)
        course_id = ingest_syllabus(app.state.store, app.state.llm, file.filename, text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except LLMError as e:
        raise HTTPException(502, f"Extraction failed: {e}")
    return next(c for c in app.state.store.courses() if c["id"] == course_id)


@app.delete("/courses/{course_id}", status_code=204)
def delete_course(course_id: int):
    if not app.state.store.delete_course(course_id):
        raise HTTPException(404, "Course not found.")


@app.get("/deadlines", response_model=list[DeadlineOut])
def deadlines(start: Optional[date] = None, end: Optional[date] = None,
              course: Optional[list[str]] = Query(None), kind: Optional[list[str]] = Query(None)):
    return app.state.store.deadlines(start and start.isoformat(), end and end.isoformat(), course, kind)


@app.get("/calendar.ics")
def calendar(course: Optional[list[str]] = Query(None)):
    ics = to_ics(app.state.store.deadlines(courses=course))
    return Response(ics, media_type="text/calendar",
                    headers={"Content-Disposition": 'attachment; filename="coursekit-deadlines.ics"'})


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    return answer_question(req.question, app.state.store, app.state.llm, today(), k=app.state.settings.top_k)
