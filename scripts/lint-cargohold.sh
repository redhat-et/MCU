#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../cargohold"

exec golangci-lint run \
  --timeout=5m \
  ./...
