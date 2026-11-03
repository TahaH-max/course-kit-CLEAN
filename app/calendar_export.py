"""Export deadlines as an iCalendar (.ics) file that Google Calendar, Outlook, and Apple Calendar import."""

import hashlib
from datetime import date, datetime, timedelta, timezone


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def to_ics(deadlines: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//CourseKit//Deadlines//EN", "CALSCALE:GREGORIAN"]
    for d in deadlines:
        day = date.fromisoformat(d["date"])
        # Stable UID: re-importing updates the event instead of duplicating it.
        uid = hashlib.sha1(f"{d['course']}|{d['title']}".encode()).hexdigest()[:16]
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}@coursekit",
            f"DTSTAMP:{stamp}",
            f"DTSTART;VALUE=DATE:{day:%Y%m%d}",
            f"DTEND;VALUE=DATE:{day + timedelta(days=1):%Y%m%d}",
            f"SUMMARY:{_escape(d['course'] + ': ' + d['title'])}",
            f"DESCRIPTION:{_escape('From syllabus: ' + d['source_line'])}",
            f"CATEGORIES:{d['kind'].upper()}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines) + "\r\n"
