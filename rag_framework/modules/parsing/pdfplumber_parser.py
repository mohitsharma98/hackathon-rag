"""
Local PDF parser using pdfplumber.

Supports local paths and Databricks / cloud storage URIs:
  - dbfs:/...        → copied via Databricks dbutils
  - abfss://...      → copied via Databricks dbutils
  - wasbs://...      → copied via Databricks dbutils
  - /dbfs/...        → used directly (already a local FUSE mount path)
  - any other path   → treated as a local filesystem path
"""

import shutil
import tempfile
import time

from rag_framework.config.models import ParserConfig
from rag_framework.core.exceptions import ParsingError
from rag_framework.core.interfaces import BaseParser, ParsedDocument

# URI schemes that require downloading to a local temp file before parsing
_REMOTE_SCHEMES = ("dbfs:/", "abfss://", "wasbs://", "abfs://", "s3://", "gs://")


class PDFPlumberParser(BaseParser):
    """
    Parses PDF files locally using pdfplumber.

    Accepts local paths and Databricks/cloud storage URIs.
    Remote files are copied to a temp directory first via dbutils (on Databricks)
    or the standard urllib for other schemes.

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

        local_path, _tmp_dir = self._resolve_local_path(file_path)

        t0 = time.perf_counter()
        pages_text: list[str] = []

        try:
            with pdfplumber.open(local_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    pages_text.append(text)
        except Exception as e:
            raise ParsingError(f"pdfplumber failed on '{file_path}': {e}") from e
        finally:
            if _tmp_dir:
                shutil.rmtree(_tmp_dir, ignore_errors=True)

        full_text = "\n\n".join(pages_text)
        elapsed = time.perf_counter() - t0

        return ParsedDocument(
            text=full_text,
            pages=pages_text,
            metadata={
                "source": file_path,
                "parser": "pdfplumber",
                "page_count": page_count,
                "parse_time_s": round(elapsed, 3),
                "char_count": len(full_text),
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_local_path(file_path: str) -> tuple[str, str | None]:
        """
        Return (local_path, tmp_dir_to_cleanup).

        If file_path is already local, returns (file_path, None).
        If it's a remote/DBFS URI, copies it to a temp dir and returns
        (temp_local_path, tmp_dir).
        """
        # /dbfs/ is the FUSE mount — directly accessible as a local path
        if file_path.startswith("/dbfs/"):
            return file_path, None

        if not any(file_path.startswith(scheme) for scheme in _REMOTE_SCHEMES):
            return file_path, None

        tmp_dir = tempfile.mkdtemp()
        filename = file_path.split("/")[-1] or "file.pdf"
        local_path = f"{tmp_dir}/{filename}"

        if file_path.startswith(("dbfs:/", "abfss://", "wasbs://", "abfs://")):
            # Use Databricks dbutils — available in notebook and job contexts
            try:
                from pyspark.dbutils import DBUtils  # noqa: F401
                from pyspark.sql import SparkSession
                spark = SparkSession.getActiveSession()
                if spark is None:
                    raise RuntimeError("No active SparkSession.")
                dbutils = DBUtils(spark)
                dbutils.fs.cp(file_path, f"file:{local_path}")
            except Exception as e:
                raise ParsingError(
                    f"Failed to copy '{file_path}' via dbutils: {e}. "
                    "Ensure this runs inside a Databricks cluster."
                ) from e
        else:
            # Generic fallback for s3://, gs://, etc.
            try:
                import urllib.request
                urllib.request.urlretrieve(file_path, local_path)
            except Exception as e:
                raise ParsingError(
                    f"Failed to download '{file_path}': {e}"
                ) from e

        return local_path, tmp_dir
