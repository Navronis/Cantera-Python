#!/bin/bash
# Diagnose stale wheel: timestamps, embedded numpy ABI, and venv contents.
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src
ls -la --time-style=full-iso build/python/dist/*.whl
ls -la --time-style=full-iso build/python/cantera/_cantera*.so
"$venv/python" -c "import cantera, os; print(cantera.__file__)"
ls -la --time-style=full-iso /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so 2>/dev/null
# What numpy does the compiled extension expect vs. provide?
strings build/python/cantera/_cantera*.so | grep -m3 'numpy\.'
strings /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so 2>/dev/null | grep -m3 'numpy\.'
