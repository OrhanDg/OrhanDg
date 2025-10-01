"""Utilities for extracting and normalising resume and job description text."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import docx2txt
from pypdf import PdfReader


SUPPORTED_FILE_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "text/plain": ".txt",
}


class UnsupportedFileTypeError(ValueError):
    """Raised when a supplied file does not match the supported formats."""


def _read_pdf(file_obj: BinaryIO) -> str:
    reader = PdfReader(file_obj)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def _read_docx(file_obj: BinaryIO) -> str:
    with BytesIO(file_obj.read()) as buffer:
        buffer.seek(0)
        text = docx2txt.process(buffer)
    return text.strip()


def _read_txt(file_obj: BinaryIO) -> str:
    return file_obj.read().decode("utf-8", errors="ignore").strip()


def extract_text_from_upload(file_name: str, file_bytes: bytes, mime_type: str | None = None) -> str:
    """Extract text from an uploaded resume file.

    Parameters
    ----------
    file_name:
        The original name of the uploaded file.
    file_bytes:
        Raw bytes of the uploaded file.
    mime_type:
        Optional mime type provided by Streamlit. When absent, it is
        inferred from the file extension.
    """

    extension = (SUPPORTED_FILE_TYPES.get(mime_type) if mime_type else Path(file_name).suffix).lower()
    reader: BinaryIO

    if extension not in {".pdf", ".docx", ".doc", ".txt"}:
        raise UnsupportedFileTypeError(
            "Unsupported resume format. Please upload a PDF, DOCX, DOC or TXT file."
        )

    if extension == ".pdf":
        reader = BytesIO(file_bytes)
        return _read_pdf(reader)

    if extension in {".docx", ".doc"}:
        reader = BytesIO(file_bytes)
        return _read_docx(reader)

    # Default to plain text
    reader = BytesIO(file_bytes)
    return _read_txt(reader)


def normalise_job_description(text: str) -> str:
    """Normalise whitespace and ensure a clean job description string."""
    cleaned = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    return cleaned
