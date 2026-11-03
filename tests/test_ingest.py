import pytest

from app.ingest import chunk_text, extract_text


def test_chunks_carry_heading_path():
    text = "# Warranty\n\n## Deductible\n\nA $100 deductible applies per visit."
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0].section == "Warranty > Deductible"
    assert chunks[0].text.startswith("Warranty > Deductible\n")


def test_long_sections_are_split_under_word_limit():
    para = " ".join(["word"] * 60)
    text = "# Policy\n\n" + "\n\n".join([para] * 5)
    chunks = chunk_text(text, max_words=140)
    assert len(chunks) == 3  # 2 + 2 + 1 paragraphs
    assert all(len(c.text.split()) <= 140 + 2 for c in chunks)


def test_sibling_headings_reset_path():
    text = "# A\n\n## B\n\none\n\n## C\n\ntwo"
    assert [c.section for c in chunk_text(text)] == ["A > B", "A > C"]


def test_unsupported_file_type_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_text("notes.docx", b"hello")


def test_plain_text_decodes():
    assert extract_text("a.txt", "Loaner policy".encode()) == "Loaner policy"


def test_real_syllabus_chunks_carry_course_and_section():
    from .conftest import sample_text
    chunks = chunk_text(sample_text("csc216_data_structures.md"))
    late = next(c for c in chunks if c.section.endswith("Late Work Policy"))
    assert late.section == "CSC 216: Data Structures > Late Work Policy"
    assert "10% per day late" in late.text
    assert all(len(c.text.split()) <= 160 for c in chunks)  # 140-word limit + heading line