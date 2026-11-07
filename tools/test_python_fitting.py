#!/usr/bin/env python3
"""Test Python GasTransport polynomial fitting vs native Cantera."""
import sys
import math
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cantera_python as ct
from cantera_python.transport import (
    _astar_table, _bstar_table, _cstar_table, _omega22_table,
    GasTransportData, Frot
)
from cantera_python.constants import (
    Boltzmann, GasConstant, Avogadro, Pi, epsilon_0, lightSpeed
)

# Load GRI-3.0 in pure Python
gas = ct.Solution("gri30.yaml")
thermo = gas.thermo
nsp = thermo.n_species

print(f"Loaded mechanism with {nsp} species.")
print("Min temp:", thermo.min_temp, "Max temp:", thermo.max_temp)

# Let's inspect species 0 (H2) transport data
tdata_0 = gas.transport_data.get("H2") if hasattr(gas, "transport_data") else None
print("H2 transport data:", tdata_0)
