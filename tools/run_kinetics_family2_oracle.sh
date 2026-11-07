#!/usr/bin/env bash
set -e
cd /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python

export CANTERA_DATA="/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data:/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/data:/home/nebula/cantera-pinned-oracle/data"

./tools/run_native_kinetics_family2_oracle audit/corpus_kinetics_family2.jsonl audit/native_kinetics_family2_oracle_results.json
