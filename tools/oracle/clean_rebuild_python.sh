#!/bin/bash
# Clean rebuild of oracle Python bindings, explicitly against the venv
# interpreter (previous rebuild picked up system numpy 1.26.4).
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src

"$venv/python" -c "import numpy; print('venv numpy:', numpy.__version__)"

rm -rf build/python build/temp-py

scons build python_cmd="$venv/python" -j4 2>&1 | grep -iE "NumPy version|python interpreter|done building|error" | head -8

wheel=$(ls -t build/python/dist/*.whl | head -1)
echo "installing: $wheel"
"$venv/pip" install --quiet --force-reinstall --no-deps "$wheel"

"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/diag_oracle_reactor2.py
