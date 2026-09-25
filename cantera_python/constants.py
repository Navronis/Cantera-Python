"""Physical/numerical constants from Cantera include/cantera/base/ct_defs.h.

Units are m, kg, s, K, kmol. Derived equations retain the pinned source's
operation order. Original copyright and BSD terms are in ../License.txt.
"""
from __future__ import annotations
import math

Pi = 3.14159265358979323846
Sqrt2 = 1.41421356237309504880
Avogadro = 6.02214076e26
Boltzmann = 1.380649e-23
Planck = 6.62607015e-34
ElectronCharge = 1.602176634e-19
lightSpeed = 299792458.0
OneAtm = 1.01325e5
OneBar = 1.0e5
fineStructureConstant = 7.2973525693e-3
ElectronMass = 9.1093837015e-31
GasConstant = Avogadro * Boltzmann
logGasConstant = math.log(GasConstant)
GasConst_cal_mol_K = GasConstant / 4184.0
StefanBoltz = (2.0 * math.pow(Pi, 5) * math.pow(Boltzmann, 4)
              / (15.0 * math.pow(Planck, 3) * lightSpeed * lightSpeed))
Faraday = ElectronCharge * Avogadro
permeability_0 = 2 * fineStructureConstant * Planck / (ElectronCharge * ElectronCharge * lightSpeed)
epsilon_0 = 1.0 / (lightSpeed * lightSpeed * permeability_0)
SmallNumber = 1.0e-300
BigNumber = 1.0e300
Undef = -999.1234
Tiny = 1.0e-20

# Upstream equilibrium constraint identifiers, not implementations of solvers.
TV, HP, SP, PV, TP, UV, ST, SV, UP, VH, TH, SH, PX, TX = range(100, 114)
VT, PH, PS, VP, PT, VU, TS, VS, PU, HV, HT, HS, XP, XT = range(-100, -114, -1)


class CanteraError(ValueError):
    """Base exception for Cantera errors."""
    _stack_trace_depth = 20

    def __init__(self, *args):
        super().__init__(*args)
        self._msg = str(args[0]) if args else "Cantera error"
        self._method = str(args[0]) if len(args) > 1 else ""

    def __str__(self):
        if len(self.args) == 2:
            return f"\nCanteraError thrown by {self.args[0]}:\n{self.args[1]}"
        elif len(self.args) == 1:
            return str(self.args[0])
        return super().__str__()

    def CanteraError(self) -> CanteraError:
        return self

    def what(self) -> str:
        msg = str(self)
        return msg

    def getMessage(self) -> str:
        msg = str(self)
        return msg

    def getMethod(self) -> str:
        method_name = self._method if self._method else self.__class__.__name__
        return method_name

    def getClass(self) -> str:
        cls_name = self.__class__.__name__
        return cls_name

    @classmethod
    def setStackTraceDepth(cls, depth: int) -> None:
        cls._stack_trace_depth = int(depth)

class ArraySizeError(CanteraError):
    def getMessage(self) -> str:
        msg = str(self)
        return msg

    def getClass(self) -> str:
        cls_name = "ArraySizeError"
        return cls_name

class IndexError(CanteraError):
    def getMessage(self) -> str: return str(self)
    def getClass(self) -> str: return "IndexError"

class InputFileError(CanteraError):
    def getClass(self) -> str: return "InputFileError"

class NotImplementedError(CanteraError):
    def getClass(self) -> str: return "NotImplementedError"
