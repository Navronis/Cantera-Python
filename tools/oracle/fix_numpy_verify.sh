#!/bin/bash
# Align venv numpy with build ABI, verify oracle import + reactor integration.
set -e
venv=/tmp/cantera-oracle/venv/bin
"$venv/pip" install --quiet --force-reinstall 'numpy==2.5.3'
"$venv/python" -c "import numpy; print('numpy', numpy.__version__)"
"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/diag_oracle_reactor2.py
