#!/usr/bin/env python3
"""
Quick local verification script for Azure Document Intelligence.

Usage:
    # Uses env vars AZURE_DI_ENDPOINT and AZURE_DI_KEY
    python scripts/verify_azure_di.py path/to/file.pdf

    # Or pass credentials explicitly
    AZURE_DI_ENDPOINT=https://... AZURE_DI_KEY=... python scripts/verify_azure_di.py file.pdf
"""

import os
import sys

def main():
    if len(sys.argv) < 2:
        # Use a small inline text snippet via a temp file if no PDF provided
        import tempfile, textwrap
        sample_text = textwrap.dedent("""\
            Hello from Azure Document Intelligence verification script.
            This is a test document with some sample text.
        """)
        tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp.write(sample_text.encode())
        tmp.close()
        file_path = tmp.name
        print(f"No file provided — using temp file: {file_path}")
    else:
        file_path = sys.argv[1]

    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    api_key  = os.getenv("AZURE_DI_KEY")

    if not endpoint or not api_key:
        print("ERROR: Set AZURE_DI_ENDPOINT and AZURE_DI_KEY env vars before running.")
        sys.exit(1)

    print(f"Endpoint : {endpoint}")
    print(f"File     : {file_path}")
    print(f"Model    : prebuilt-document")
    print("Connecting to Azure Document Intelligence...")

    from rag_framework.config.models import ParserConfig, ParserImpl
    from rag_framework.modules.parsing.azure_di_parser import AzureDIParser

    config = ParserConfig(
        implementation=ParserImpl.azure_di,
        azure_endpoint=endpoint,
        azure_api_key=api_key,
        azure_model_id="prebuilt-document",
    )
    parser = AzureDIParser(config)

    # Health check first
    try:
        parser.health_check()
        print("Health check passed.")
    except Exception as e:
        print(f"Health check FAILED: {e}")
        sys.exit(1)

    # Parse the document
    try:
        doc = parser.parse(file_path)
    except Exception as e:
        print(f"Parsing FAILED: {e}")
        sys.exit(1)

    print("\n--- Result ---")
    print(f"Pages      : {doc.metadata.get('page_count')}")
    print(f"Characters : {doc.metadata.get('char_count')}")
    print(f"Parse time : {doc.metadata.get('parse_time_s')}s")
    print(f"\nText preview (first 500 chars):\n{doc.text[:500]}")
    print("\nAzure Document Intelligence is working correctly.")


if __name__ == "__main__":
    main()
