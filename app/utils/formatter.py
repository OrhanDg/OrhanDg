"""Formatting utilities for ATS friendly resumes and cover letters."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Iterable, List, Sequence

from docx import Document


@dataclass
class ResumeSection:
    title: str
    lines: Sequence[str]


def _add_heading(document: Document, text: str, level: int = 1) -> None:
    heading = document.add_paragraph()
    run = heading.add_run(text.upper())
    run.bold = True
    if level == 1:
        run.font.size = heading.style.font.size


def _add_bullets(document: Document, lines: Iterable[str]) -> None:
    for line in lines:
        if not line:
            continue
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.add_run(line)


def resume_to_docx(full_name: str, contact_block: Sequence[str], sections: Sequence[ResumeSection]) -> bytes:
    """Render an ATS friendly resume into a DOCX document."""
    document = Document()

    title = document.add_heading(level=0)
    title.alignment = 1
    title_run = title.add_run(full_name)
    title_run.bold = True

    contact_paragraph = document.add_paragraph()
    contact_paragraph.alignment = 1
    contact_paragraph.add_run(" | ".join(contact_block))

    for section in sections:
        _add_heading(document, section.title)
        _add_bullets(document, section.lines)

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.read()


def cover_letter_to_docx(subject: str, paragraphs: Sequence[str], closing: str) -> bytes:
    document = Document()

    if subject:
        heading = document.add_heading(level=1)
        heading_run = heading.add_run(subject)
        heading_run.bold = True

    for paragraph in paragraphs:
        doc_paragraph = document.add_paragraph()
        doc_paragraph.add_run(paragraph)

    footer = document.add_paragraph()
    footer.add_run(closing)

    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.read()


def plain_text_resume(full_name: str, contact_block: Sequence[str], sections: Sequence[ResumeSection]) -> str:
    blocks: List[str] = [full_name.upper(), " | ".join(contact_block)]
    for section in sections:
        blocks.append(section.title.upper())
        blocks.extend(f"- {line}" for line in section.lines if line)
    return "\n\n".join(blocks)


def plain_text_cover_letter(subject: str, paragraphs: Sequence[str], closing: str) -> str:
    blocks = [subject] if subject else []
    blocks.extend(paragraphs)
    blocks.append(closing)
    return "\n\n".join(blocks)
