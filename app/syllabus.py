"""Syllabus -> structured course data (deadlines, grading weights), verified against the text.

Extraction can come from the LLM or from the offline rule-based parser. Either
way the result goes through the same checks before it is saved:

- every deadline must quote a `source_line` that really appears in the syllabus
- the deadline's date must be a date actually written on that line
- grading weights must appear in the syllabus and should add up to 100%
- deadlines should fall inside the term

Deadlines that fail the first two checks are dropped (and reported); the rest
are kept but produce warnings. A wrong due date is worse than a missing one.
"""

import re
from datetime import date
from typing import Optional

from .llm import ChatLLM
from .schemas import Deadline, GradeComponent, SyllabusExtract

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
_MONTH_RE = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE_PATTERNS = [
    re.compile(rf"\b{_MONTH_RE}\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?", re.I),
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    re.compile(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b"),
]

EXTRACT_PROMPT = """Extract structured data from this course syllabus.
Rules:
- course_code like "CSC 216". Dates as YYYY-MM-DD; infer the year from the term.
- deadlines: only graded items with a date (assignments, quizzes, exams, papers). Skip plain lecture topics.
- For each deadline, source_line must be the syllabus line copied EXACTLY.
- grading: each component and its percent weight exactly as written. Do not fix or normalize weights."""


# ---------- dates ----------

def find_dates(text: str, default_year: int) -> list[date]:
    found: list[date] = []
    for m in _DATE_PATTERNS[0].finditer(text):
        month = MONTHS[m.group(1).lower()[:3]]
        year = int(m.group(3)) if m.group(3) else default_year
        found += _safe_date(year, month, int(m.group(2)))
    for m in _DATE_PATTERNS[1].finditer(text):
        found += _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    for m in _DATE_PATTERNS[2].finditer(text):
        year = int(m.group(3)) if m.group(3) else default_year
        year = year + 2000 if year < 100 else year
        found += _safe_date(year, int(m.group(1)), int(m.group(2)))
    return found


def _safe_date(y: int, m: int, d: int) -> list[date]:
    try:
        return [date(y, m, d)]
    except ValueError:
        return []


def classify(title: str) -> str:
    t = title.lower()
    deliverable = r"\b(assignment|homework|hw|project|lab|essay|paper|report|draft|proposal|review|document)\b"
    if re.search(r"\bexam\b", t) or re.search(r"\bmidterm\b", t) and not re.search(deliverable, t):
        return "exam"
    if re.search(r"\bquiz", t):
        return "quiz"
    if re.search(deliverable, t) or re.search(r"\bdue\b", t):
        return "assignment"  # checked before bare "final" so "Final report" isn't an exam
    if re.search(r"\bfinal\b", t):
        return "exam"
    return "other"


# ---------- offline parser ----------

def _field(text: str, label: str) -> Optional[str]:
    m = re.search(rf"^{label}\s*:\s*(.+)$", text, re.I | re.M)
    return m.group(1).strip() if m else None


def _section(text: str, name: str) -> str:
    m = re.search(rf"^#+\s*{name}[^\n]*\n(.*?)(?=^#+\s|\Z)", text, re.I | re.M | re.S)
    return m.group(1) if m else ""


def _term(text: str) -> tuple[Optional[date], Optional[date], int]:
    term = _field(text, "Term") or ""
    year_m = re.search(r"\b(20\d\d)\b", term)
    year = int(year_m.group(1)) if year_m else date.today().year
    dates = find_dates(term, year)
    if len(dates) >= 2:
        return dates[0], dates[1], year
    return None, None, year


def parse_offline(text: str) -> SyllabusExtract:
    course = _field(text, "Course") or ""
    code_m = re.search(r"\b([A-Z]{2,4})\s?(\d{3}[A-Z]?)\b", course or text)
    code = f"{code_m.group(1)} {code_m.group(2)}" if code_m else "UNKNOWN"
    course_title = course.split("-", 1)[1].strip() if "-" in course else None

    start, end, year = _term(text)

    grading = []
    for line in _section(text, "Grading").splitlines():
        m = re.match(r"\s*(?:[-*]|\|)\s*([^|:]+?)\s*(?::|\|)\s*(\d+(?:\.\d+)?)\s*%", line)
        if m and m.group(1).strip().lower() not in {"component"}:
            grading.append(GradeComponent(component=m.group(1).strip(), weight=float(m.group(2))))

    deadlines = []
    for line in _section(text, "Schedule").splitlines():
        stripped = line.strip()
        if not stripped or set(stripped) <= set("|-: "):
            continue
        dates = find_dates(stripped, year)
        if not dates:
            continue
        # title = text after the date, minus separators
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) >= 2:
            title = cells[1]
        else:
            title = re.split(r"\d{1,2}(?:st|nd|rd|th)?\s*[:—–-]\s*", stripped, maxsplit=1)[-1]
        title = title.strip(" -:")
        kind = classify(title)
        if kind == "other":
            continue
        d = dates[0]
        if start and end and d < start and end.year > start.year:
            d = d.replace(year=end.year)
        deadlines.append(Deadline(title=title, date=d.isoformat(), kind=kind, source_line=stripped))

    return SyllabusExtract(
        course_code=code,
        title=course_title,
        instructor=_field(text, "Instructor"),
        term_start=start.isoformat() if start else None,
        term_end=end.isoformat() if end else None,
        grading=grading,
        deadlines=deadlines,
    )


# ---------- verification ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("—", "-").replace("–", "-")).strip().lower()


def verify(extract: SyllabusExtract, text: str) -> tuple[SyllabusExtract, list[str]]:
    warnings: list[str] = []
    doc = _norm(text)
    _, _, year = _term(text)
    start = date.fromisoformat(extract.term_start) if extract.term_start else None
    end = date.fromisoformat(extract.term_end) if extract.term_end else None

    kept: list[Deadline] = []
    seen = set()
    for d in extract.deadlines:
        if not d.source_line or _norm(d.source_line) not in doc:
            warnings.append(f"Dropped '{d.title}': its source line isn't in the syllabus")
            continue
        try:
            due = date.fromisoformat(d.date)
        except ValueError:
            warnings.append(f"Dropped '{d.title}': invalid date '{d.date}'")
            continue
        line_dates = find_dates(d.source_line, year)
        if not any((ld.month, ld.day) == (due.month, due.day) for ld in line_dates):
            warnings.append(f"Dropped '{d.title}': {d.date} isn't the date written on its line")
            continue
        key = (d.title.lower(), d.date)
        if key in seen:
            continue
        seen.add(key)
        if start and end and not (start <= due <= end):
            warnings.append(f"'{d.title}' on {d.date} falls outside the term")
        kept.append(d)

    grading_text = _norm(_section(text, "Grading") or text)
    for g in extract.grading:
        w = f"{g.weight:g}%"
        if w not in grading_text.replace(" %", "%"):
            warnings.append(f"Grading weight {w} for '{g.component}' doesn't appear in the syllabus")
    total = sum(g.weight for g in extract.grading)
    if extract.grading and abs(total - 100) > 0.5:
        warnings.append(f"Grading weights add up to {total:g}%, not 100%; check with the instructor")
    if not extract.grading:
        warnings.append("No grading breakdown found")

    return extract.model_copy(update={"deadlines": kept}), warnings


def extract_syllabus(text: str, llm: Optional[ChatLLM]) -> tuple[SyllabusExtract, list[str]]:
    raw = parse_offline(text) if llm is None else llm.structured(EXTRACT_PROMPT, text, SyllabusExtract)
    return verify(raw, text)
