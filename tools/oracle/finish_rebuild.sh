#!/bin/bash
# Resume/finish the oracle build with bundled SUNDIALS 7.5, install, verify.
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src

scons build \
    python_cmd="$venv/python" \
    sundials_include=/tmp/cantera-oracle/sundials-install/include \
    sundials_libdir=/tmp/cantera-oracle/sundials-install/lib \
    -j4 2>&1 | tail -6

ls -la build/python/dist/
wheel=$(ls -t build/python/dist/*.whl | head -1)
echo "installing: $wheel"
"$venv/pip" install --quiet --force-reinstall --no-deps "$wheel"

echo "--- linked sundials:"
ldd /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so | grep -i sundials || echo "none dynamic"

"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/diag_oracle_reactor2.py
