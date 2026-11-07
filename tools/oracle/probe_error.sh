#!/bin/bash
# Capture the full scons python-probe error.
cd /tmp/cantera-oracle/src
venv=/tmp/cantera-oracle/venv/bin
scons build prefix=/tmp/cantera-oracle/install python_cmd="$venv/python" -j4 2>&1 | head -60
