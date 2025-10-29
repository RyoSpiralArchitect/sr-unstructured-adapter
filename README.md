# SR Unstructured Adapter

Turn chaotic files into a predictable JSON payload that downstream tools love.

## Features
- Normalises paths, MIME types, and metadata for any file.
- Extracts clean text and rich metadata from HTML, DOCX, PDF, XLSX, and email sources.
- Summarises archives, logs, CSV/TSV sheets, and image scans while surfacing attachment metadata.
- Pretty prints JSON sources and captures schema hints (top-level keys, type).
- Captures text statistics such as line, character, and word counts.
- Generates chat-ready message chunks for LLM ingestion.
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

Use JSON lines mode to stream results to other processes:

```bash
python -m sr_adapter.adapter --as-json-lines examples/sample.txt
```

## Library usage
```python
from sr_adapter import build_payload, to_llm_messages

payload = build_payload("examples/sample.txt")
print(payload.to_dict())

messages = to_llm_messages(payload, chunk_size=1000)
```

## Development
Run the tests with `pytest`:

```bash
pytest
```
