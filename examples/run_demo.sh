#!/usr/bin/env bash
set -euo pipefail

python -m sr_adapter.adapter --as-json-lines examples/sample.txt
