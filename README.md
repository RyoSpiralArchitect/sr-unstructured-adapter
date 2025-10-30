# SR Unstructured Adapter

Turn chaotic files into a predictable JSON payload that downstream tools love.

## Features
- Normalises paths, MIME types, and metadata for any file.
- Extracts clean text and rich metadata from HTML, DOCX, PDF, RTF, PPTX, XLSX, and email sources.
- Summarises archives, logs (with severity/timestamp rollups), CSV/TSV sheets, and image scans while surfacing attachment metadata.
- Pretty prints JSON sources and captures schema hints (top-level keys, type).
- Captures text statistics such as line, character, and word counts.
- Generates chat-ready message chunks and Markdown bundles optimised for LLM ingestion.
- Normalises output into a unified document schema with doc-level confidence,
  provenance, and validation hints.
- Provides a simple CLI (`python -m sr_adapter.adapter`) that can emit either
  JSON lines or a formatted list.

## Installation
This repository uses a [PEP 621](https://peps.python.org/pep-0621/) compatible
`pyproject.toml`. Install in editable mode while iterating:

```bash
pip install -e .
```

## Quickstart
```bash
python -m sr_adapter.adapter examples/sample.txt
```

### Output
```json
{
  "schema_version": "1.0",
  "doc_id": "5f77e23c-...",
  "doc_type": "text",
  "source": "/absolute/path/to/examples/sample.txt",
  "mime": "text/plain",
  "confidence": 0.67,
  "text_blocks": [
    {
      "block_index": 0,
      "type": "paragraph",
      "text": "...",
      "confidence": 0.5
    }
  ],
  "tables": [],
  "items": [],
  "parties": [],
  "amounts": [],
  "dates": [
    {
      "value": "2025-01-01",
      "origin": "meta.modified_at",
      "confidence": 0.6
    }
  ],
  "attachments": [],
  "meta": {
    "size": 123,
    "line_count": 10,
    "char_count": 118,
    "word_count": 24,
    "modified_at": "2025-01-01T00:00:00+00:00"
  },
  "provenance": {
    "parties": [],
    "amounts": [],
    "dates": [
      {
        "value": "2025-01-01",
        "origin": "meta.modified_at",
        "block_index": null
      }
    ],
    "items": []
  },
  "validation": {
    "warnings": [],
    "errors": []
  },
  "highlights": {
    "summary": "First few lines of the document…",
    "key_points": ["Key date 2025-01-01"]
  },
  "llm_ready": {
    "markdown": "# Document: Text...",
    "sections": [
      {
        "title": "Chunk 1",
        "text": "First chunk of text..."
      }
    ],
    "chunks": [
      {
        "title": "Chunk 1",
        "text": "First chunk of text...",
        "approx_tokens": 128
      }
    ]
  }
}
```

The raw `build_payload` API remains available if you only need lightweight
metadata:

```python
from sr_adapter import build_payload

payload = build_payload("examples/sample.txt")
print(payload.to_dict())
```

Log inputs expose additional metadata including `log_line_count`,
severity-ordered `log_levels`, aggregated `log_level_counts`, example messages,
and first/last timestamps while the unified payload emits `log` blocks and
validation warnings when high-severity entries are present.

Use JSON lines mode to stream results to other processes:

```bash
python -m sr_adapter.adapter --as-json-lines examples/sample.txt
```

## Library usage
```python
from sr_adapter import build_llm_bundle, build_payload, to_llm_messages, to_unified_payload

payload = build_payload("examples/sample.txt")
messages = to_llm_messages(payload, chunk_size=1000)
unified = to_unified_payload(payload)
print(unified["highlights"])
markdown_bundle = build_llm_bundle(unified)
```

## Development
Run the tests with `pytest`:

```bash
pytest
```
