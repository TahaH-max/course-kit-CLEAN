import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.embeddings import HashingEmbedder
from app.ingest import chunk_text
from app.llm import ChatLLM
from app.store import Store
from app.syllabus import extract_syllabus

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "sample_syllabi"
TODAY = date(2026, 11, 3)  # a Tuesday, mid-semester


def sample_text(name: str) -> str:
    return (SAMPLES / name).read_text()


@pytest.fixture
def store():
    s = Store(":memory:", HashingEmbedder())
    for path in sorted(SAMPLES.glob("*.md")):
        text = path.read_text()
        extract, warnings = extract_syllabus(text, None)
        s.add_syllabus(path.name, extract, warnings, chunk_text(text))
    return s


class FakeOpenAIClient:
    """Mimics client.chat.completions.create with scripted replies, so the real
    ChatLLM validation/retry code runs with no network."""

    def __init__(self, replies):
        self.replies = [r if isinstance(r, str) else json.dumps(r) for r in replies]
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.replies.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.fixture
def fake_llm():
    def make(*replies):
        client = FakeOpenAIClient(list(replies))
        return ChatLLM(client, model="fake", provider="fake"), client

    return make
