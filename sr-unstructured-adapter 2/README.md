# SR Unstructured Adapter

Convert messy/unstructured files into a **unified JSON payload** for LLMs & search.

## Quickstart
```bash
python -m sr_adapter.adapter examples/sample.txt
```

### Output (JSON)
```json
{"source":"examples/sample.txt","mime":"text/plain","text":"...","meta":{"size":123}}
```

## Supported (now)
- `.txt` / `.md` / `.json` (basic)
- `.html` / `.pdf` → TODO（stub）

## Why
Give downstream systems a *consistent* payload, even when inputs are chaotic.
