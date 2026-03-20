"""
Local PDF parser using PyMuPDF (fitz).
"""

import time

from rag_framework.config.models import ParserConfig
from rag_framework.core.exceptions import ParsingError
from rag_framework.core.interfaces import BaseParser, ParsedDocument


class PyMuPDFParser(BaseParser):
    """
    Parses PDF files locally using PyMuPDF (fitz).

    Strengths : very fast, handles images, good plain-text extraction.
    Limitation: table extraction requires extra work.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def health_check(self) -> None:
        """Verify pymupdf is importable."""
        try:
            import fitz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is not installed. Run: pip install pymupdf"
            ) from e

    def parse(self, file_path: str) -> ParsedDocument:
        """Extract text page-by-page using PyMuPDF."""
        try:
            import fitz
        except ImportError as e:
            raise ImportError("PyMuPDF is not installed.") from e

        t0 = time.perf_counter()
        pages_text: list[str] = []

        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page in doc:
                pages_text.append(page.get_text())
            doc.close()
        except Exception as e:
            raise ParsingError(f"PyMuPDF failed on '{file_path}': {e}") from e

        full_text = "\n\n".join(pages_text)
        elapsed = time.perf_counter() - t0

        return ParsedDocument(
            text=full_text,
            metadata={
                "source": file_path,
                "parser": "pymupdf",
                "page_count": page_count,
                "parse_time_s": round(elapsed, 3),
                "char_count": len(full_text),
            },
        )
