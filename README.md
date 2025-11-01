<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
# SR Unstructured Adapter

Turn chaotic documents into structured payloads with a pipeline that speaks both native kernels and LLMs.

## Why this adapter?
- **Streaming document pipeline** – Detects formats, parses into blocks, normalises text, and applies recipes without loading whole archives into memory. 【F:src/sr_adapter/pipeline.py†L1-L155】【F:src/sr_adapter/normalize.py†L1-L140】
- **Native acceleration** – Visual layout and text normalisation are executed by C++ kernels orchestrated through a shared runtime for deterministic telemetry and warm-up. 【F:src/sr_adapter/runtime.py†L1-L211】
- **LLM escalation built-in** – A driver manager routes low-confidence spans to Azure OpenAI, OpenAI, Anthropic, Docker, or local vLLM endpoints, then normalises responses into a consistent schema. 【F:src/sr_adapter/delegate.py†L1-L120】【F:src/sr_adapter/drivers/manager.py†L1-L160】【F:src/sr_adapter/drivers/openai_driver.py†L1-L80】【F:src/sr_adapter/drivers/anthropic_driver.py†L1-L80】【F:src/sr_adapter/drivers/vllm_driver.py†L1-L80】【F:src/sr_adapter/normalizer/llm_normalizer.py†L1-L120】
- **Config-first ergonomics** – Recipes describe parsing behaviour, while tenant and adapter YAML plus `.env` overrides keep credentials and runtime toggles out of code. 【F:configs/tenants/default.yaml†L1-L10】【F:configs/settings.yaml†L1-L13】【F:src/sr_adapter/settings.py†L1-L150】
- **Observability ready** – Kernel telemetry can be exported as Prometheus metrics or pushed to Sentry directly from the CLI. 【F:src/sr_adapter/telemetry.py†L1-L150】【F:src/sr_adapter/cli.py†L1-L360】

## Architecture at a glance
```
  +---------------------------+
  |  Input sources            |
  |  (files, streams)         |
  +-------------+-------------+
                |
                v
       +--------+---------+
       | Type + MIME      |
       | detection        |
       +--------+---------+
                |
                v
       +--------+---------+
       | Parser registry  |
       +--------+---------+
                |
                v
       +--------+---------+
       | Parsed blocks    |
       +--------+---------+
                |
                v
       +--------+---------+
       | NativeKernelRuntime|
       | (text & layout     |
       |  kernels)          |
       +--------+---------+
                |
                v
       +--------+---------+
       | Recipe transforms |
       +---+-----------+---+
           |           |
           |           v
           |   +-------+--------+
           |   | Confidence      |
           |   | check           |
           |   +---+--------+----+
           |       |        |
           |   High|        |Low
           |       v        v
           |   +---+----+  +---------------+
           |   | Writers |  | DriverManager|
           |   | (JSONL/ |  | (Azure,      |
           |   |  API)   |  |  Docker, …)  |
           |   +---+----+  +-------+-------+
           |                        |
           |                        v
           |               +--------+--------+
           |               | LLM drivers     |
           |               +--------+--------+
           |                        |
           |                        v
           |               +--------+--------+
           |               | LLM normaliser  |
           |               +--------+--------+
           |                        |
           +------------------------+
                (enriched blocks)

       .-------------------------------------------.
       |  VisualLayoutAnalyzer (calibration store)  |
       '-------------------------+-----------------'
                                 |
                                 v
       +-------------------------+-----------------+
       |     NativeKernelRuntime (shared state)     |
       +-------------------------------------------+
```

## Key components
### Parser + recipe stack
1. `ParserRegistry` detects a best-fit parser by MIME or sniffed type and streams heavyweight formats (PDFs, images) chunk-by-chunk. 【F:src/sr_adapter/pipeline.py†L11-L153】
2. Each block is normalised, enriched by the active recipe, and written to structured documents or downstream sinks. 【F:src/sr_adapter/pipeline.py†L155-L320】

### Native kernels runtime
- `NativeKernelRuntime` coordinates the C++ text normaliser and layout analyser, capturing per-kernel telemetry and offering fast warm-up hooks. 【F:src/sr_adapter/runtime.py†L38-L207】
- Layout calibration persists between runs so repeated profiles skip warm-up and reuse tuned thresholds. 【F:src/sr_adapter/visual.py†L1-L220】
- Set `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` to fall back to the pure Python path or adjust batching with `SR_ADAPTER_TEXT_KERNEL_BATCH_BYTES`. 【F:src/sr_adapter/runtime.py†L213-L244】【F:src/sr_adapter/normalize.py†L103-L140】

### Processing profiles
- Processing profiles bundle runtime layout preferences and LLM escalation policy so UX stays simple while the system adapts to each workload. The registry exposes built-ins (`balanced`, `realtime`, `archival`) and loads overrides from `configs/profiles/` or custom search paths. 【F:src/sr_adapter/profiles.py†L1-L215】【F:configs/profiles/balanced.yaml†L1-L18】
- The `PipelineOrchestrator` resolves the active profile, warms the runtime when requested, and only escalates blocks that satisfy the profile's confidence, type, and limit criteria. 【F:src/sr_adapter/pipeline.py†L1-L360】
- CLI commands accept `--profile` so you can swap latency vs. fidelity trade-offs without changing recipes or code. 【F:src/sr_adapter/cli.py†L19-L80】【F:src/sr_adapter/pipeline.py†L320-L420】

### LLM escalation
- `delegate.escalate_low_conf` loads the configured recipe, resolves the tenant, and invokes the appropriate driver through the shared manager cache. 【F:src/sr_adapter/delegate.py†L1-L120】
- Drivers live in `src/sr_adapter/drivers/` and register themselves with a lightweight factory registry, so dropping in Azure, OpenAI, Anthropic, Docker, or vLLM backends requires no manager changes. 【F:src/sr_adapter/drivers/base.py†L1-L170】【F:src/sr_adapter/drivers/azure_driver.py†L1-L200】【F:src/sr_adapter/drivers/openai_driver.py†L1-L80】【F:src/sr_adapter/drivers/anthropic_driver.py†L1-L80】【F:src/sr_adapter/drivers/vllm_driver.py†L1-L80】
- Responses are normalised into a stable schema before the pipeline writes them back into documents or CLI output. 【F:src/sr_adapter/normalizer/llm_normalizer.py†L1-L120】

## Installation
Install the adapter in editable mode while iterating:

```bash
pip install -e .
```

Optional native builds will be triggered automatically when the kernels are first imported; ensure a C++17 toolchain is available.

## Configuration
### Recipes
Recipes live under `src/sr_adapter/recipes` and control parser options, confidence thresholds, and escalation toggles.

### LLM tenants
1. Copy `.env.llm.example` to `.env` (or export the variables manually). 【F:.env.llm.example†L1-L9】
2. Drop tenant YAML into `configs/tenants/`. The `driver` key selects a backend and `settings` contain endpoint-specific knobs. 【F:configs/tenants/default.yaml†L1-L10】
3. Reference tenants from recipes via the `llm.tenant` field or override at runtime with `SR_ADAPTER_TENANT`.

### Adapter settings
1. Global runtime defaults live in `configs/settings.yaml` (telemetry, driver defaults, distributed backends). 【F:configs/settings.yaml†L1-L13】
2. `sr_adapter.settings.get_settings()` merges YAML with environment overrides and `.env` values using Pydantic validation. 【F:src/sr_adapter/settings.py†L1-L150】
3. Override individual knobs via environment variables such as `SR_ADAPTER_DISTRIBUTED__DEFAULT_BACKEND=asyncio` or `SR_ADAPTER_DRIVERS__DEFAULT_TIMEOUT=45`. 【F:src/sr_adapter/settings.py†L1-L150】

### Environment toggles
- `SR_ADAPTER_DISABLE_NATIVE_RUNTIME=1` – force the legacy Python normalisers. 【F:src/sr_adapter/runtime.py†L213-L244】
- `SR_ADAPTER_TEXT_KERNEL_BATCH_BYTES=<bytes>` – cap payload size per native call. 【F:src/sr_adapter/normalize.py†L103-L140】
- `SR_ADAPTER_MAX_SIZE_MB=<float>` – guardrails for the classic adapter CLI. 【F:src/sr_adapter/adapter.py†L12-L70】

## CLI quickstart
All orchestration commands live under `python -m sr_adapter.cli`.

### Convert documents
```bash
python -m sr_adapter.cli convert docs/*.pdf --recipe default --out output.jsonl --profile balanced --backend threadpool --concurrency 4
```
Stream parsed blocks into JSONL while optionally disabling escalation with `--no-llm`. Select another processing profile (e.g. `realtime`) to trade accuracy for latency, and choose a distributed backend (`threadpool`, `asyncio`, `dask`, `ray`) when scaling batch jobs. 【F:src/sr_adapter/cli.py†L19-L110】【F:src/sr_adapter/pipeline.py†L320-L420】

### Inspect LLM drivers
```bash
# List configured tenants
python -m sr_adapter.cli llm list-tenants

# Run a single prompt with inline metadata
python -m sr_adapter.cli llm run --tenant default --prompt "Summarise this" --metadata '{"source": "demo"}'

# Replay a JSONL dataset and capture responses
python -m sr_adapter.cli llm replay --input data/escalation_samples.jsonl --output replay.jsonl --skip-errors
```
The CLI validates prompts, streams normalized responses, and can skip failures while reporting a summary. 【F:src/sr_adapter/cli.py†L85-L210】

### Manage native kernels
```bash
# Show runtime status with telemetry
python -m sr_adapter.cli kernels status

# Compile and warm both kernels, emitting JSON
python -m sr_adapter.cli kernels warm --json

# Export Prometheus metrics (and optionally send to Sentry)
python -m sr_adapter.cli kernels export --format prometheus --label env=prod --sentry
```
Status, warm-up, and exports reuse the cached runtime so repeated invocations stay snappy while still emitting observability signals. 【F:src/sr_adapter/cli.py†L19-L360】【F:src/sr_adapter/telemetry.py†L1-L150】

## Library usage
Use the high-level helpers when embedding the adapter in another service:

```python
from sr_adapter.pipeline import batch_convert

documents = batch_convert(["examples/sample.txt"], recipe="default")
for document in documents:
    print(document.to_dict())
```
`batch_convert` applies detection, parsing, native normalisation, recipes, and escalation before returning structured documents. 【F:src/sr_adapter/pipeline.py†L155-L320】

## Sample data
The `data/escalation_samples.jsonl` file provides quick prompts for replay testing. 【F:data/escalation_samples.jsonl†L1-L15】

## Development
Run the full test suite with:

```bash
pytest -q
```

The suite covers driver management, pipeline behaviours, native kernel orchestration, and CLI workflows. 【F:tests/test_cli_llm.py†L1-L395】【F:tests/test_runtime.py†L1-L44】
