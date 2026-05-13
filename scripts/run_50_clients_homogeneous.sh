#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m runner.run_experiment --num-clients 50 --rounds 100 --save-every-round true --eval-every-round true
