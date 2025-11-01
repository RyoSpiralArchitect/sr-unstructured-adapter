#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
set -euo pipefail

python -m sr_adapter.adapter --as-json-lines examples/sample.txt
