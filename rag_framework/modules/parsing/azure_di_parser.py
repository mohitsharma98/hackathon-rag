"""
Cloud PDF parser using Azure Document Intelligence.
"""

import os
import site
import sys
import time
from collections import defaultdict


def _fix_azure_namespace() -> None:
    """Extend the azure namespace path to include all site-packages directories.

    Databricks cluster libraries are installed in a separate site-packages
    directory that Python's import system discovers correctly, but the ``azure``
    namespace package may have already been initialised from the default
    Databricks environment before the cluster-library path was added to
    ``sys.path``.  When that happens ``azure.ai`` is not found even though the
    package is physically present.  Patching ``azure.__path__`` fixes it.
    """
    if "azure" not in sys.modules:
        return
    import azure  # noqa: PLC0415
    existing = set(azure.__path__)
    for sp in site.getsitepackages():
        candidate = os.path.join(sp, "azure")
        if os.path.isdir(candidate) and candidate not in existing:
            azure.__path__.append(candidate)
            existing.add(candidate)

from rag_framework.config.models import ParserConfig
from rag_framework.core.exceptions import BackendConnectionError, MissingCredentialError, ParsingError
from rag_framework.core.interfaces import BaseParser, ParsedDocument


class AzureDIParser(BaseParser):
    """
    Parses documents using Azure Document Intelligence (formerly Form Recognizer).

    Strengths : handles complex layouts, tables, handwriting, scanned docs.
    Requires  : AZURE_DI_ENDPOINT and AZURE_DI_KEY env vars (or config).
    """

    def __init__(self, config: ParserConfig):
        self.config = config
        self.endpoint = config.azure_endpoint or os.getenv("AZURE_DI_ENDPOINT")
        self.api_key = config.azure_api_key or os.getenv("AZURE_DI_KEY")
        self.model_id = config.azure_model_id

    def health_check(self) -> None:
        """Validate credentials and SDK availability."""
        if not self.endpoint:
            raise MissingCredentialError("AZURE_DI_ENDPOINT")
        if not self.api_key:
            raise MissingCredentialError("AZURE_DI_KEY")
        _fix_azure_namespace()
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient  # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"azure-ai-documentintelligence is not importable ({e}). "
                "If the package is installed, try: pip install --upgrade --force-reinstall azure-ai-documentintelligence"
            ) from e
        # TODO: make a lightweight connectivity check (e.g. list models) to confirm auth

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Submit the document to Azure Document Intelligence and extract text.

        TODO: Add support for table extraction and structured output.
        """
        self.health_check()

        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            from azure.core.credentials import AzureKeyCredential
        except ImportError as e:
            raise ImportError("azure-ai-documentintelligence not installed.") from e

        t0 = time.perf_counter()

        # /dbfs/Volumes/... is not a valid FUSE path for Unity Catalog Volumes.
        # The correct path is /Volumes/... (no /dbfs/ prefix).
        local_path = file_path
        if local_path.startswith("/dbfs/Volumes/"):
            local_path = local_path[len("/dbfs"):]

        try:
            with open(local_path, "rb") as f:
                file_bytes = f.read()
        except OSError as e:
            raise ParsingError(f"Cannot read file '{local_path}': {e}") from e

        try:
            client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key),
            )
            import io
            poller = client.begin_analyze_document(
                model_id=self.model_id,
                body=io.BytesIO(file_bytes),
                content_type="application/octet-stream",
            )
            result = poller.result()
        except Exception as e:
            if "401" in str(e) or "403" in str(e):
                raise BackendConnectionError("Azure Document Intelligence", f"Auth failed: {e}") from e
            raise ParsingError(f"Azure DI failed on '{file_path}': {e}") from e

        page_count = len(result.pages) if result.pages else 0

        # Group paragraphs by page number to populate per-page text
        page_texts: dict[int, list[str]] = defaultdict(list)
        for paragraph in (result.paragraphs or []):
            page_num = (
                paragraph.bounding_regions[0].page_number
                if paragraph.bounding_regions else 1
            )
            page_texts[page_num].append(paragraph.content)

        pages = ["\n\n".join(page_texts[i]) for i in range(1, page_count + 1)]
        full_text = "\n\n".join(p for p in pages if p)

        # Extract tables
        tables = []
        for table in (result.tables or []):
            page_num = (
                table.bounding_regions[0].page_number
                if table.bounding_regions else 1
            )
            grid = [[""] * table.column_count for _ in range(table.row_count)]
            for cell in (table.cells or []):
                grid[cell.row_index][cell.column_index] = cell.content

            md_rows = []
            for r, row in enumerate(grid):
                md_rows.append("| " + " | ".join(row) + " |")
                if r == 0:
                    md_rows.append("| " + " | ".join(["---"] * table.column_count) + " |")

            tables.append({
                "page_number": page_num,
                "row_count": table.row_count,
                "column_count": table.column_count,
                "data": grid,
                "markdown": "\n".join(md_rows),
            })

        elapsed = time.perf_counter() - t0

        return ParsedDocument(
            text=full_text,
            pages=pages,
            tables=tables,
            metadata={
                "source": file_path,
                "parser": "azure_di",
                "model_id": self.model_id,
                "page_count": page_count,
                "parse_time_s": round(elapsed, 3),
                "char_count": len(full_text),
            },
        )
