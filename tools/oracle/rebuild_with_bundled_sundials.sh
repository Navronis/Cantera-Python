#!/bin/bash
# Rebuild the pinned oracle with its own bundled SUNDIALS (submodule) instead
# of the system installation, to fix the frozen-reactor integration issue.
set -e
venv=/tmp/cantera-oracle/venv/bin
cd /tmp/cantera-oracle/src

# Build the pinned SUNDIALS submodule first
cd /tmp/cantera-oracle/src/ext/sundials
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/tmp/cantera-oracle/sundials-install \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_KLU=OFF 2>&1 | tail -3
make -j4 2>&1 | tail -2
make install 2>&1 | tail -2
echo "--- sundials installed"

# Reconfigure Cantera against it
cd /tmp/cantera-oracle/src
rm -f cantera.conf
scons config sundials_include=/tmp/cantera-oracle/sundials-install/include \
    sundials_libdir=/tmp/cantera-oracle/sundials-install/lib \
    prefix=/tmp/cantera-oracle/install python_cmd="$venv/python" 2>&1 | grep -iE "sundials|blas|lapack" | head -10

scons build -j4 2>&1 | tail -5

"$venv/pip" install --quiet --force-reinstall --no-deps \
    build/python/dist/Cantera-*.whl build/python/dist/cantera-*.whl 2>/dev/null \
    || "$venv/pip" install --quiet --force-reinstall --no-deps build/python/dist/*antera*.whl

"$venv/python" -c "import cantera as ct; print('OK', ct.__version__, ct.__git_commit__)"
