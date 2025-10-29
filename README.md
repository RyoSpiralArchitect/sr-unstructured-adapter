# SR Unstructured Adapter

Turn chaotic files into a predictable JSON payload that downstream tools love.

## Features
- Normalises paths, MIME types, and metadata for any file.
- Pretty prints JSON sources and captures schema hints (top-level keys, type).
- Captures text statistics such as line, character, and word counts.
- Generates chat-ready message chunks for LLM ingestion.
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
[
  {
    "source": "/absolute/path/to/examples/sample.txt",
    "mime": "text/plain",
    "text": "...",
    "meta": {
      "size": 123,
      "line_count": 10,
      "char_count": 118,
      "modified_at": "2025-01-01T00:00:00+00:00"
    }
  }
]
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
