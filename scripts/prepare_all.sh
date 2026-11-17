#!/usr/bin/env bash
set -euo pipefail
python3 -m ricce.cli prepare --config configs/project.yaml
pytest -q
