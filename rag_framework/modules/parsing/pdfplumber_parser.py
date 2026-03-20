"""
Local PDF parser using pdfplumber.
"""

import time

from rag_framework.config.models import ParserConfig
from rag_framework.core.exceptions import ParsingError
from rag_framework.core.interfaces import BaseParser, ParsedDocument


class PDFPlumberParser(BaseParser):
    """
    Parses PDF files locally using pdfplumber.

    Strengths : handles complex layouts, tables, multi-column text.
    Limitation: slower than PyMuPDF on large files.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def health_check(self) -> None:
        """Verify pdfplumber is importable."""
        try:
            import pdfplumber  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "pdfplumber is not installed. Run: pip install pdfplumber"
            ) from e

    def parse(self, file_path: str) -> ParsedDocument:
        """Extract text from all pages of a PDF."""
        try:
            import pdfplumber
        except ImportError as e:
            raise ImportError("pdfplumber is not installed.") from e

        t0 = time.perf_counter()
        pages_text: list[str] = []

        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
        except Exception as e:
            raise ParsingError(f"pdfplumber failed on '{file_path}': {e}") from e

        full_text = "\n\n".join(pages_text)
        elapsed = time.perf_counter() - t0

        return ParsedDocument(
            text=full_text,
            metadata={
                "source": file_path,
                "parser": "pdfplumber",
                "page_count": page_count,
                "parse_time_s": round(elapsed, 3),
                "char_count": len(full_text),
            },
        )
