#!/bin/bash
# Run the pinned-oracle verification script against the venv.
set -e
venv=/tmp/cantera-oracle/venv/bin
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/verify_oracle.py
