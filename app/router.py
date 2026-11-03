"""Decide how to answer a question, and turn it into structured filters.

Three routes:
- "deadlines": time questions ("what's due next week?", "when is midterm 2?") -> SQL over extracted deadlines
- "grading":   weight questions ("how much is the final worth?")               -> SQL over grading table
- "policy":    everything else ("can I bring a calculator?")                   -> retrieval + verified LLM answer

Why route at all: "what's due next week" needs date arithmetic across every
course, which similarity search over text chunks is bad at. The structured path
is exact, fast, and needs no LLM. Routing is rule-based so it's deterministic
and every decision can be explained; an LLM classifier is a drop-in upgrade.
"""

import calendar
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from .syllabus import find_dates

TEMPORAL = re.compile(
    r"\b(due|deadlines?|when|upcoming|coming up|schedule|next|this week|tomorrow|today|dates?|"
    r"what do i have|this month|before|by (?:mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)|"
    r"in (?:january|february|march|april|may|june|july|august|september|october|november|december))\b",
    re.I,
)
# Specific item words narrow results: "the final" should not also return the midterm.
ITEM_WORDS = ["final", "midterm", "homework", "quiz", "participation", "project", "proposal",
              "report", "draft", "peer review", "lab", "essay", "paper", "attendance"]
GRADING = re.compile(r"\b(worth|weight(?:ed)?|percent(?:age)?|grade breakdown|grading breakdown|count(?:s)? for)\b|%", re.I)
KIND_WORDS = {
    "exam": r"\b(exams?|midterms?|finals?)\b(?!\s+(report|paper|project|draft))",
    "quiz": r"\bquiz(zes)?\b",
    "assignment": r"\b(assignments?|homeworks?|hw|projects?|papers?|essays?|reports?|proposals?|peer reviews?|labs?)\b",
}
ITEM_RE = re.compile(
    r"\b(midterm|quiz|homework|hw|assignment|programming assignment|peer review|exam)\s*#?\s*(\d+)\b", re.I)
GENERIC_TITLE_WORDS = {"introduction", "intro", "course", "principles", "fundamentals"}
MONTH_NAMES = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}


@dataclass
class Route:
    kind: str
    courses: list[str] = field(default_factory=list)
    start: Optional[date] = None
    end: Optional[date] = None
    kinds: list[str] = field(default_factory=list)
    item: Optional[tuple[str, str]] = None   # e.g. ("midterm", "2")
    words: list[str] = field(default_factory=list)  # e.g. ["final"]: titles must contain all of them
    first_only: bool = False                 # "when is my next exam?"

    def matches(self, title: str) -> bool:
        t = title.lower()
        if "draft" in t and "draft" not in self.words:
            return False  # "final report" means the report, not the draft
        return all(w in t for w in self.words)

    def explain(self) -> str:
        parts = [f"route={self.kind}"]
        if self.courses:
            parts.append(f"courses={','.join(self.courses)}")
        if self.start or self.end:
            parts.append(f"dates={self.start or '…'}..{self.end or '…'}")
        if self.kinds:
            parts.append(f"kinds={','.join(self.kinds)}")
        if self.item:
            parts.append(f"item={' '.join(self.item)}")
        elif self.words:
            parts.append(f"title contains {'+'.join(self.words)}")
        if self.first_only:
            parts.append("first upcoming only")
        return " · ".join(parts)


def detect_courses(question: str, course_index: list[dict]) -> list[str]:
    q = question.lower()
    found = []
    for c in course_index:
        code = c["code"].lower()
        dept, num = code.split() if " " in code else (code, "")
        if re.search(rf"\b{dept}\s?{num}\b", q) or (num and re.search(rf"\b{num}\b", q)):
            found.append(c["code"])
            continue
        words = [w for w in re.findall(r"[a-z]+", (c.get("title") or "").lower())
                 if len(w) > 4 and w not in GENERIC_TITLE_WORDS]
        if any(re.search(rf"\b{w}\b", q) for w in words):
            found.append(c["code"])
    return found


def strip_course_mentions(question: str, courses: list[str], course_index: list[dict]) -> str:
    """Remove course codes/titles from a query once they've been applied as a filter.

    Every chunk carries its course name in the heading path, so leaving "CSC 216" in the
    query makes every chunk of that course look relevant and defeats the retrieval gate.
    """
    q = question
    for c in course_index:
        if c["code"] not in courses:
            continue
        dept, _, num = c["code"].partition(" ")
        q = re.sub(rf"\b{dept}\s?{num}\b|\b{num}\b", " ", q, flags=re.I)
        for w in re.findall(r"[A-Za-z]+", c.get("title") or ""):
            if len(w) > 4 and w.lower() not in GENERIC_TITLE_WORDS:
                q = re.sub(rf"\b{w}\b", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


def parse_range(question: str, today: date) -> tuple[Optional[date], Optional[date]]:
    q = question.lower()
    if "tomorrow" in q:
        d = today + timedelta(days=1)
        return d, d
    if re.search(r"\btoday\b", q):
        return today, today
    if "next week" in q:
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6)
    if "this week" in q:
        return today, today + timedelta(days=6 - today.weekday())
    if m := re.search(r"next (\d+) days", q):
        return today, today + timedelta(days=int(m.group(1)))
    if "this month" in q:
        return today, today.replace(day=calendar.monthrange(today.year, today.month)[1])
    for name, num in MONTH_NAMES.items():
        if re.search(rf"\bin {name}\b", q):
            year = today.year if num >= today.month else today.year + 1
            return date(year, num, 1), date(year, num, calendar.monthrange(year, num)[1])
    if m := re.search(r"\b(?:before|by)\s+(.+)", q):
        dates = find_dates(m.group(1), today.year)
        if dates:
            return today, dates[0]
    return None, None


def item_words(question: str) -> list[str]:
    q = question.lower()
    return [w for w in ITEM_WORDS if re.search(rf"\b{w}(?:s|zes|es)?\b", q)]


def route_question(question: str, today: date, course_index: list[dict]) -> Route:
    courses = detect_courses(question, course_index)
    kinds = [k for k, pat in KIND_WORDS.items() if re.search(pat, question, re.I)]
    words = item_words(question)

    if GRADING.search(question) and not re.search(r"\b(due|when)\b", question, re.I):
        return Route(kind="grading", courses=courses, words=words)

    if not TEMPORAL.search(question):
        return Route(kind="policy", courses=courses)

    item = ITEM_RE.search(question)
    if re.match(r"\s*when\s+(is|are|do|does)\b", question, re.I) and not (kinds or words or item):
        # "When is spring break?" / "When are office hours?" names something that isn't a
        # graded deadline, so search the syllabus text instead of listing deadlines.
        return Route(kind="policy", courses=courses)

    route = Route(kind="deadlines", courses=courses, kinds=kinds)
    if item:
        route.item = (item.group(1).lower(), item.group(2))
        return route  # a specific named item: search all dates
    route.words = words
    route.start, route.end = parse_range(question, today)
    if route.start is None:
        if re.search(r"\bnext\b", question, re.I) and kinds:
            route.start, route.first_only = today, True       # "when is my next quiz?"
        elif re.search(r"\b(final|when)\b", question, re.I) and kinds:
            route.start = None                                 # "when is the final?": any date
        else:
            route.start, route.end = today, today + timedelta(days=14)  # default: next two weeks
    return route
