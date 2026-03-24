"""
Local PDF parser using pdfplumber.

Supports local paths and Databricks / cloud storage URIs:
  - /dbfs/...        → used directly via FUSE mount
  - dbfs:/...        → read into memory via Hadoop FS (no temp files)
  - abfss://...      → read into memory via Hadoop FS (no temp files)
  - wasbs://...      → read into memory via Hadoop FS (no temp files)
  - any other path   → treated as a local filesystem path
"""

import io
import time

from rag_framework.config.models import ParserConfig
from rag_framework.core.exceptions import ParsingError
from rag_framework.core.interfaces import BaseParser, ParsedDocument

_REMOTE_SCHEMES = ("dbfs:/", "abfss://", "wasbs://", "abfs://", "s3://", "gs://")


class PDFPlumberParser(BaseParser):
    """
    Parses PDF files using pdfplumber.

    Accepts local paths and Databricks/cloud storage URIs.
    Remote files are read directly into memory (BytesIO) — no temp files or
    filesystem copies required.

    Strengths : handles complex layouts, tables, multi-column text.
    Limitation: slower than PyMuPDF on large files.
    """

    def __init__(self, config: ParserConfig):
        self.config = config

    def health_check(self) -> None:
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

        file_obj = self._open(file_path)

        t0 = time.perf_counter()
        pages_text: list[str] = []

        try:
            with pdfplumber.open(file_obj) as pdf:
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
    def _open(file_path: str) -> str | io.BytesIO:
        """
        Return a local path string or an in-memory BytesIO for remote URIs.

        /dbfs/ paths are returned as-is (FUSE mount, already local).
        All other remote URIs are read into BytesIO via the Hadoop FS API
        — no temp files or filesystem copies needed.
        """
        # FUSE mount — directly accessible as a normal local path
        if file_path.startswith("/dbfs/"):
            return file_path

        # Plain local path
        if not any(file_path.startswith(scheme) for scheme in _REMOTE_SCHEMES):
            return file_path

        # Remote URI — read bytes into memory via Hadoop FS (Databricks)
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.getActiveSession()
            if spark is None:
                raise RuntimeError("No active SparkSession.")
            hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
            uri = spark.sparkContext._jvm.java.net.URI(file_path)
            fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(uri, hadoop_conf)
            path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(file_path)
            stream = fs.open(path)
            byte_array = spark.sparkContext._jvm.org.apache.commons.io.IOUtils.toByteArray(stream)
            stream.close()
            return io.BytesIO(bytes(byte_array))
        except Exception as e:
            raise ParsingError(
                f"Failed to read '{file_path}' via Hadoop FS: {e}. "
                "Ensure this runs inside a Databricks cluster."
            ) from e
