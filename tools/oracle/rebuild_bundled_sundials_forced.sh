#!/bin/bash
# Rebuild oracle forcing the bundled SUNDIALS 7.5 install, then verify
# (a) which sundials the extension links, (b) reactor integration works.
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src

rm -rf build/python build/temp-py build/src
rm -f cantera.conf

scons build \
    python_cmd="$venv/python" \
    sundials_include=/tmp/cantera-oracle/sundials-install/include \
    sundials_libdir=/tmp/cantera-oracle/sundials-install/lib \
    -j4 2>&1 | grep -iE "sundials|NumPy version|done building|error" | head -12

wheel=$(ls -t build/python/dist/*.whl | head -1)
echo "installing: $wheel"
"$venv/pip" install --quiet --force-reinstall --no-deps "$wheel"

echo "--- linked sundials:"
ldd /tmp/cantera-oracle/venv/lib/python3.12/site-packages/cantera/_cantera*.so | grep -i sundials || echo "static/no dynamic sundials"

"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
"$venv/python" /mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/tools/oracle/diag_oracle_reactor2.py
