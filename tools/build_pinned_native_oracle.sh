#!/usr/bin/env bash
# Reproducible build script for Cantera exact pinned commit: 726522be4e2a13454d8415b7ef799d621f665cf3
set -euo pipefail

PINNED_COMMIT="726522be4e2a13454d8415b7ef799d621f665cf3"
SRC_DIR="${1:-$HOME/cantera-pinned-oracle}"
VENV_DIR="${2:-$HOME/cantera-oracle-env}"

echo "=== Verifying Pinned Commit ==="
if [ ! -d "$SRC_DIR" ]; then
    echo "Cloning source repository..."
    git clone https://github.com/Cantera/cantera.git "$SRC_DIR"
fi

cd "$SRC_DIR"
git fetch --all --tags
git checkout "$PINNED_COMMIT"
git submodule update --init --recursive

ACTUAL_COMMIT=$(git rev-parse HEAD)
echo "Active commit: $ACTUAL_COMMIT"
if [ "$ACTUAL_COMMIT" != "$PINNED_COMMIT" ]; then
    echo "ERROR: Active commit $ACTUAL_COMMIT does not match required $PINNED_COMMIT" >&2
    exit 1
fi

echo "=== Setting up Build Environment ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install scons numpy cython scipy ruamel.yaml pyyaml pytest

echo "=== Building Cantera Native Library ==="
cat << 'EOF' > cantera.conf
prefix = '/usr/local/cantera-pinned'
python_package = 'y'
f90_interface = 'n'
system_sundials = 'n'
cxx_flags = '-O3 -std=c++20'
EOF

python3 -m SCons build -j$(nproc)
python3 -m SCons install

echo "=== Native Oracle Smoke Test ==="
python3 -c "
import cantera as ct
print('Cantera version:', ct.__version__)
print('Git commit:', getattr(ct, '__git_commit__', 'N/A'))
gas = ct.Solution('gri30.yaml')
gas.TP = 500.0, 200000.0
gas.HP = gas.enthalpy_mass, gas.P
assert abs(gas.T - 500.0) < 1e-4, 'HP solve failed'
print('Native build verified successfully!')
"

echo "=== Build Complete ==="
