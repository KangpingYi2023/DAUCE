#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec "${FACTORJOIN_PYTHON:-python}" "$ROOT/scripts/figure8.py" "$@"
