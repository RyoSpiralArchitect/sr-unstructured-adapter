# SR Unstructured Adapter

Turn chaotic files into a predictable JSON payload that downstream tools love.

## Features
- Normalises paths, MIME types, and metadata for any file.
- Extracts clean text and rich metadata from HTML, DOCX, PDF, PPTX, XLSX, XML, YAML, and email sources.
- Summarises archives, logs (with severity/timestamp rollups), CSV/TSV sheets, and image scans while surfacing attachment metadata.
- Pretty prints JSON sources and captures schema hints (top-level keys, type).
- Captures text statistics such as line, character, and word counts.
- Generates chat-ready message chunks for LLM ingestion with metadata-prefixed context.
- Emits `llm_hints` and a richer `llm` bundle (`brief`, `focus`, `outline`) so prompts can highlight key facts without heavy prompting.
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
  "llm": {
    "brief": "Text prepared for downstream LLM consumption. Key facts: Word count: 24 words.",
    "focus": [
      {"label": "word_count", "summary": "Approximate length: 24 words", "confidence": 0.55, "source": "meta.word_count"}
    ],
    "outline": [
      {"title": null, "kind": "paragraph", "preview": "Paragraph: lorem ipsum…", "block_indices": [0], "confidence": 0.5}
    ],
    "hints": [
      "Detected type: text",
      "Word count: 24",
      "Line count: 10"
    ]
  },
  "llm_hints": [
    "Detected type: text",
    "Word count: 24",
    "Line count: 10"
  ],
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

Log inputs expose additional metadata including `log_line_count`,
severity-ordered `log_levels`, aggregated `log_level_counts`, example messages,
and first/last timestamps while the unified payload emits `log` blocks,
validation warnings when high-severity entries are present, and `llm_hints`
summarising counts and severity.

Use JSON lines mode to stream results to other processes:

```bash
python -m sr_adapter.adapter --as-json-lines examples/sample.txt
```

For richer control, the CLI entry-point supports writing raw documents, base
payloads, or fully unified payloads. Use ``sr_adapter.cli`` to batch-convert
files, stream to ``stdout`` or disk, and tolerate individual failures:

```bash
python -m sr_adapter.cli convert examples/ --format unified --out out.jsonl
python -m sr_adapter.cli convert input.txt --format payload --out -
```

When an error occurs, conversion continues and an error record is emitted in the
output stream; add ``--strict`` to surface a non-zero exit code instead.

## Library usage
```python
from sr_adapter import build_payload, to_llm_messages, to_unified_payload

payload = build_payload("examples/sample.txt")
print(payload.to_dict())

# Pass unified LLM facets alongside the payload for richer prompts
unified = to_unified_payload("examples/sample.txt")
payload_dict = {**payload.to_dict(), "llm": unified.get("llm", {})}
messages = to_llm_messages(payload_dict, chunk_size=1000)
```

## Development
Run the tests with `pytest`:

```bash
pytest
```
