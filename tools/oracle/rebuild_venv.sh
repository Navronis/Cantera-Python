#!/bin/bash
# Pin venv numpy to the exact version the build uses, rebuild, verify.
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src

"$venv/pip" install --quiet --force-reinstall 'numpy==2.5.3'
"$venv/python" -c "import numpy; print('venv numpy now:', numpy.__version__)"
"$venv/python" -c "import ctypes, numpy; print('dtype size:', ctypes.sizeof(numpy.dtype('f8')))" || true

scons build prefix=/tmp/cantera-oracle/install python_cmd="$venv/python" -j4 2>&1 | grep -E "NumPy version|done building|Error" | head -5

"$venv/pip" install --quiet --force-reinstall --no-deps \
  build/python/dist/Cantera-*.whl build/python/dist/cantera-*.whl 2>/dev/null \
  || "$venv/pip" install --quiet --force-reinstall --no-deps build/python/dist/*cantera*.whl

"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
