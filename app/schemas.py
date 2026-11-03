"""Pydantic models shared by the API, the LLM layer, and the tests."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["exam", "quiz", "assignment", "other"]


# ---------- Syllabus extraction ----------

class GradeComponent(BaseModel):
    component: str
    weight: float = Field(description="Percent of the final grade, e.g. 25 for 25%.")


class Deadline(BaseModel):
    title: str
    date: str = Field(description="ISO date YYYY-MM-DD.")
    kind: Kind = "other"
    source_line: str = Field(description="The exact line of the syllabus this came from.")


class SyllabusExtract(BaseModel):
    """What the extractor must return. Validated, then checked against the document."""

    course_code: str
    title: Optional[str] = None
    instructor: Optional[str] = None
    term_start: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    term_end: Optional[str] = Field(default=None, description="YYYY-MM-DD")
    grading: list[GradeComponent] = []
    deadlines: list[Deadline] = []


class CourseOut(BaseModel):
    id: int
    code: str
    title: Optional[str]
    instructor: Optional[str]
    document: str
    grading: list[GradeComponent]
    deadline_count: int
    warnings: list[str]


class DeadlineOut(BaseModel):
    course: str
    title: str
    date: str
    kind: Kind
    source_line: str


# ---------- Question answering ----------

class Citation(BaseModel):
    source_id: str
    quote: str


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    citations: list[Citation] = []


class SourceOut(BaseModel):
    source_id: str
    document: str
    section: str
    score: float
    text: str


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    route: Literal["deadlines", "grading", "policy"]
    status: Literal["answered", "grounded", "not_found", "unverified"]
    answer: str
    deadlines: list[DeadlineOut] = []
    grading: list[dict] = []
    citations: list[Citation] = []
    sources: list[SourceOut] = []
    checks: list[CheckResult] = []
    explanation: str = ""
