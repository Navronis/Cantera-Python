#!/bin/bash
# Install the freshly built oracle wheel (one file only) and verify.
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src
ls -la build/python/dist/
# Prefer the newest wheel by modification time
wheel=$(ls -t build/python/dist/*.whl | head -1)
echo "installing: $wheel"
"$venv/pip" install --quiet --force-reinstall --no-deps "$wheel"
"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/diag_oracle_reactor2.py
