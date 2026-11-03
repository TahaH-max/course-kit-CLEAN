"""SQLite storage.

Two kinds of data live side by side:
- structured: courses, grading weights, deadlines -> answered with SQL
- unstructured: syllabus text chunks + embedding vectors -> answered with retrieval

Uploading a syllabus writes both in one transaction, and re-uploading the same
file replaces everything that came from it.
"""

import sqlite3
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ingest import Chunk
from .schemas import SyllabusExtract

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL UNIQUE,
    title TEXT,
    instructor TEXT,
    term_start TEXT,
    term_end TEXT,
    document TEXT NOT NULL,
    warnings TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS grading (
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    component TEXT NOT NULL,
    weight REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS deadlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    due_date TEXT NOT NULL,          -- ISO date, so string comparison sorts correctly
    source_line TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deadlines_date ON deadlines(due_date);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
    section TEXT NOT NULL,
    text TEXT NOT NULL,
    embedder TEXT NOT NULL,
    vector BLOB NOT NULL
);
"""


@dataclass
class Hit:
    course: str
    document: str
    section: str
    text: str
    score: float


class Store:
    def __init__(self, db_path: str, embedder):
        self.embedder = embedder
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self._reembed_if_embedder_changed()

    # ---------- writes ----------

    def add_syllabus(self, document: str, extract: SyllabusExtract, warnings: list[str],
                     chunks: list[Chunk]) -> int:
        vectors = self.embedder.embed([c.text for c in chunks]) if chunks else []
        with self.conn:
            self.conn.execute("DELETE FROM courses WHERE code = ? OR document = ?",
                              (extract.course_code, document))
            cur = self.conn.execute(
                "INSERT INTO courses (code, title, instructor, term_start, term_end, document, warnings) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (extract.course_code, extract.title, extract.instructor, extract.term_start,
                 extract.term_end, document, "\n".join(warnings)),
            )
            cid = cur.lastrowid
            self.conn.executemany("INSERT INTO grading VALUES (?, ?, ?)",
                                  [(cid, g.component, g.weight) for g in extract.grading])
            self.conn.executemany(
                "INSERT INTO deadlines (course_id, title, kind, due_date, source_line) VALUES (?, ?, ?, ?, ?)",
                [(cid, d.title, d.kind, d.date, d.source_line) for d in extract.deadlines],
            )
            self.conn.executemany(
                "INSERT INTO chunks (course_id, section, text, embedder, vector) VALUES (?, ?, ?, ?, ?)",
                [(cid, c.section, c.text, self.embedder.name, v.astype(np.float32).tobytes())
                 for c, v in zip(chunks, vectors)],
            )
        return cid

    def delete_course(self, course_id: int) -> bool:
        with self.conn:
            return self.conn.execute("DELETE FROM courses WHERE id = ?", (course_id,)).rowcount > 0

    def _reembed_if_embedder_changed(self):
        rows = self.conn.execute("SELECT id, text FROM chunks WHERE embedder != ?",
                                 (self.embedder.name,)).fetchall()
        if rows:
            vecs = self.embedder.embed([r["text"] for r in rows])
            with self.conn:
                self.conn.executemany("UPDATE chunks SET embedder = ?, vector = ? WHERE id = ?",
                                      [(self.embedder.name, v.tobytes(), r["id"]) for r, v in zip(rows, vecs)])

    # ---------- structured reads (SQL) ----------

    def courses(self) -> list[dict]:
        out = []
        for c in self.conn.execute("SELECT * FROM courses ORDER BY code").fetchall():
            grading = [dict(component=g["component"], weight=g["weight"]) for g in
                       self.conn.execute("SELECT component, weight FROM grading WHERE course_id = ?", (c["id"],))]
            n = self.conn.execute("SELECT COUNT(*) FROM deadlines WHERE course_id = ?", (c["id"],)).fetchone()[0]
            out.append(dict(id=c["id"], code=c["code"], title=c["title"], instructor=c["instructor"],
                            document=c["document"], grading=grading, deadline_count=n,
                            warnings=[w for w in c["warnings"].split("\n") if w]))
        return out

    def deadlines(self, start: Optional[str] = None, end: Optional[str] = None,
                  courses: Optional[list[str]] = None, kinds: Optional[list[str]] = None) -> list[dict]:
        sql = ("SELECT c.code AS course, d.title, d.due_date AS date, d.kind, d.source_line "
               "FROM deadlines d JOIN courses c ON c.id = d.course_id WHERE 1=1")
        params: list = []
        if start:
            sql += " AND d.due_date >= ?"
            params.append(start)
        if end:
            sql += " AND d.due_date <= ?"
            params.append(end)
        if courses:
            sql += f" AND c.code IN ({','.join('?' * len(courses))})"
            params += courses
        if kinds:
            sql += f" AND d.kind IN ({','.join('?' * len(kinds))})"
            params += kinds
        sql += " ORDER BY d.due_date, c.code"
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def grading(self, courses: Optional[list[str]] = None) -> list[dict]:
        sql = "SELECT c.code AS course, g.component, g.weight FROM grading g JOIN courses c ON c.id = g.course_id"
        params: list = []
        if courses:
            sql += f" WHERE c.code IN ({','.join('?' * len(courses))})"
            params = list(courses)
        return [dict(r) for r in self.conn.execute(sql + " ORDER BY c.code", params).fetchall()]

    def course_index(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT code, title FROM courses").fetchall()]

    # ---------- unstructured reads (retrieval) ----------

    def search(self, query: str, k: int = 4, courses: Optional[list[str]] = None) -> list[Hit]:
        sql = ("SELECT c.code, c.document, ch.section, ch.text, ch.vector FROM chunks ch "
               "JOIN courses c ON c.id = ch.course_id")
        params: list = []
        if courses:
            sql += f" WHERE c.code IN ({','.join('?' * len(courses))})"
            params = list(courses)
        rows = self.conn.execute(sql, params).fetchall()
        if not rows:
            return []
        matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        scores = matrix @ self.embedder.embed([query])[0]
        return [Hit(course=rows[i]["code"], document=rows[i]["document"], section=rows[i]["section"],
                    text=rows[i]["text"], score=float(scores[i])) for i in np.argsort(-scores)[:k]]
