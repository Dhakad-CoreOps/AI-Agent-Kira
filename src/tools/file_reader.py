import os
import sys

from langchain_core.tools import tool

from src.logger import logging
from src.exception import CustomException

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".yaml", ".yml"}


def _read_pdf(file_path: str) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "Reading PDF files requires the 'pypdf' package. Install it with: pip install pypdf"
        ) from e

    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _read_docx(file_path: str) -> str:
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "Reading .docx files requires the 'python-docx' package. Install it with: pip install python-docx"
        ) from e

    document = Document(file_path)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


@tool
def file_reader(file_path: str) -> str:
    """Read a local document (.txt, .md, .pdf, .docx and other plain-text files)
    and return its full text content. Use this to load a candidate's resume or a
    job description from disk before analysing it."""
    try:
        logging.info(f"file_reader tool invoked for path: {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at path: {file_path}")

        extension = os.path.splitext(file_path)[1].lower()

        if extension == ".pdf":
            text = _read_pdf(file_path)
        elif extension == ".docx":
            text = _read_docx(file_path)
        elif extension in SUPPORTED_TEXT_EXTENSIONS or extension == "":
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type '{extension}' for path: {file_path}")

        logging.info(f"file_reader extracted {len(text)} characters from {file_path}")
        return text.strip()

    except Exception as e:
        logging.error(f"file_reader failed for path {file_path}: {e}")
        raise CustomException(e, sys)
