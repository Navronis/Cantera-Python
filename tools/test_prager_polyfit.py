import sys
sys.path.insert(0, '.')
import math
import numpy as np
from cantera_python.constants import (
    CanteraError, ElectronCharge, Boltzmann, GasConstant, Avogadro,
    epsilon_0, Pi, lightSpeed
)

# Test polynomial helpers
def poly5(x, c):
    return (((((c[5]*x + c[4])*x + c[3])*x + c[2])*x + c[1])*x + c[0])

def poly4(x, c):
    return ((((c[4]*x + c[3])*x + c[2])*x + c[1])*x + c[0])

temp = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0]
om11_O2 = [120.0, 107.0, 98.1, 92.1, 83.0, 77.0, 72.6, 67.9, 62.7, 59.3, 56.7, 53.8]
c_prager = np.polyfit(temp, om11_O2, 5)[::-1]
print("Prager polyfit ascending coefficients:", c_prager)
print("poly5 at 1200 K:", poly5(1200.0, c_prager))
print("Actual table at 1200 K:", 72.6)
