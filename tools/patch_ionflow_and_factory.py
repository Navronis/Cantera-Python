import re

with open("cantera_python/oned.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
target_imp = "from .constants import CanteraError, OneAtm, StefanBoltz, Undef"
new_imp = "import threading\nfrom .constants import CanteraError, OneAtm, StefanBoltz, Undef, Avogadro, ElectronCharge, epsilon_0"
assert target_imp in content, "import target not found"
content = content.replace(target_imp, new_imp, 1)

# 2. Find IonFlow stub start and MultiJac start
start_idx = content.find("class IonFlow(Flow1D):")
assert start_idx != -1, "class IonFlow not found"
end_idx = content.find("class MultiJac(SystemJacobian):")
assert end_idx != -1, "class MultiJac not found"

ionflow_and_factory_code = '''def _poly5(x: float, c: Sequence[float]) -> float:
    """Evaluate polynomial of degree 5 with coefficients c[0]..c[5] in Horner form."""
    return (((((c[5] * x + c[4]) * x + c[3]) * x + c[2]) * x + c[1]) * x + c[0])


class IonFlow(Flow1D):
    """1D reacting flow domain with charged species, ion mobility, and Poisson electric field coupling.

    Direct reconstruction of Cantera's native IonFlow class.
    """

    _domain_type = "IonFlow"

    def __init__(self, phase: Solution, id: str = "flame", points: int = 1):
        super().__init__(phase, id=id, points=points)
        self.m_speciesCharge: list[float] = [float(self.gas.charge(k)) for k in range(self.m_nsp)]
        self.m_kCharge: list[int] = [k for k in range(self.m_nsp) if self.m_speciesCharge[k] != 0.0]
        self.m_kNeutral: list[int] = [k for k in range(self.m_nsp) if self.m_speciesCharge[k] == 0.0]

        # Find index of electron if present
        try:
            self.m_kElectron: Optional[int] = self.gas.species_index("E")
        except (KeyError, ValueError, CanteraError):
            self.m_kElectron = None

        # Electric potential bounds: -1e20 to 1e20
        self.setBounds(c_offset_E, -1.0e20, 1.0e20)

        # Set tighter negative species limit on charged species to avoid instabilities
        for k in self.m_kCharge:
            self.setBounds(c_offset_Y + k, -1.0e-10, 1.0)
        if self.m_kElectron is not None:
            self.setBounds(c_offset_Y + self.m_kElectron, -1.0e-14, 1.0)

        self.refiner().setActive(c_offset_E, False)
        self.m_mobility = np.zeros((self.m_points, self.m_nsp), dtype=float)
        self.m_do_electric_field: bool = False
        self.m_import_electron_transport: bool = False
        self.m_mobi_e_fix = np.zeros(6, dtype=float)
        self.m_diff_e_fix = np.zeros(6, dtype=float)

        # Validate transport model
        trans_model = getattr(self.gas.transport, "transportModel", getattr(self.gas.transport, "transport_model", ""))
        if not trans_model or trans_model == "none":
            raise CanteraError("IonFlow::IonFlow",
                "An appropriate transport model\\nshould be set when instantiating the Solution ('gas') object.")

    def domainType(self) -> str:
        if self.m_isFree:
            return "free-ion-flow"
        if self.m_usesLambda:
            return "axisymmetric-ion-flow"
        return "unstrained-ion-flow"

    def resize(self, components: int, points: int):
        super().resize(components, points)
        self.m_mobility = np.zeros((self.m_points, self.m_nsp), dtype=float)

    def componentActive(self, n: int) -> bool:
        if n == c_offset_E:
            return True
        return super().componentActive(n)

    def solveElectricField(self):
        if not self.m_do_electric_field:
            self.needJacUpdate()
        self.refiner().setActive(c_offset_U, True)
        self.refiner().setActive(c_offset_V, True)
        self.refiner().setActive(c_offset_T, True)
        self.refiner().setActive(c_offset_E, True)
        self.m_do_electric_field = True

    def fixElectricField(self):
        if self.m_do_electric_field:
            self.needJacUpdate()
        self.refiner().setActive(c_offset_U, False)
        self.refiner().setActive(c_offset_V, False)
        self.refiner().setActive(c_offset_T, False)
        self.refiner().setActive(c_offset_E, False)
        self.m_do_electric_field = False

    def doElectricField(self) -> bool:
        return self.m_do_electric_field

    def analyticJacobianSupported(self) -> bool:
        return False

    def setElectronTransport(self, tfix: Sequence[float], diff_e: Sequence[float], mobi_e: Sequence[float]):
        self.m_import_electron_transport = True
        tlog = np.log(np.asarray(tfix, dtype=float))
        self.m_diff_e_fix = np.polyfit(tlog, np.asarray(diff_e, dtype=float), 5)[::-1]
        self.m_mobi_e_fix = np.polyfit(tlog, np.asarray(mobi_e, dtype=float), 5)[::-1]

    def E(self, x: Any, j: int = 0) -> float:
        if isinstance(x, (int, np.integer)):
            j = int(x)
            return float(self.m_eField[j]) if hasattr(self, "m_eField") and len(self.m_eField) > j else 0.0
        return float(x[self.index(c_offset_E, j)])

    def dEdz(self, x: np.ndarray, j: int = 0) -> float:
        if j == 0:
            return 0.0
        return (self.E(x, j) - self.E(x, j - 1)) / (self.z(j) - self.z(j - 1))

    def ND(self, x: np.ndarray, k: int, j: int = 0) -> float:
        return Avogadro * self.m_rho[j] * self.Y(x, k, j) / self.m_wt[k]

    def rho_e(self, x: np.ndarray, j: int = 0) -> float:
        charge_density = 0.0
        for k in self.m_kCharge:
            charge_density += self.m_speciesCharge[k] * ElectronCharge * self.ND(x, k, j)
        return charge_density

    def updateTransport(self, x: np.ndarray, j0: int, j1: int) -> None:
        super().updateTransport(x, j0, j1)
        for j in range(j0, j1):
            self.setGasAtMidpoint(x, j)
            mobs = self.gas.mobilities
            if mobs is not None:
                self.m_mobility[j, :] = mobs
            if self.m_import_electron_transport and self.m_kElectron is not None:
                k = self.m_kElectron
                tlog = math.log(self.gas.T)
                self.m_mobility[j, k] = _poly5(tlog, self.m_mobi_e_fix)
                rho = self.gas.density
                wtm = self.gas.mean_molecular_weight
                self.m_diff[j, k] = self.m_wt[k] * rho * _poly5(tlog, self.m_diff_e_fix) / wtm

    def updateDiffFluxes(self, x: np.ndarray, j0: int, j1: int) -> None:
        if self.m_do_electric_field:
            self.electricFieldMethod(x, j0, j1)
        else:
            self.frozenIonMethod(x, j0, j1)

    def frozenIonMethod(self, x: np.ndarray, j0: int, j1: int) -> None:
        for j in range(j0, j1):
            dz = self.z(j + 1) - self.z(j)
            sum_neut = 0.0
            for k in self.m_kNeutral:
                self.m_flux[k, j] = self.m_diff[j, k] * (self.X(x, k, j) - self.X(x, k, j + 1)) / dz
                sum_neut -= self.m_flux[k, j]

            # correction flux to ensure that \\sum_k Y_k V_k = 0
            for k in self.m_kNeutral:
                self.m_flux[k, j] += sum_neut * self.Y(x, k, j)

            # flux for ions: zeroed out to prevent fast charged species from running away
            for k in self.m_kCharge:
                self.m_flux[k, j] = 0.0

    def electricFieldMethod(self, x: np.ndarray, j0: int, j1: int) -> None:
        for j in range(j0, j1):
            rho = self.density(j)
            dz = self.z(j + 1) - self.z(j)

            # mixture-average diffusion
            for k in range(self.m_nsp):
                self.m_flux[k, j] = self.m_diff[j, k] * (self.X(x, k, j) - self.X(x, k, j + 1)) / dz

            # ambipolar diffusion
            E_ambi = self.E(x, j)
            for k in self.m_kCharge:
                Yav = 0.5 * (self.Y(x, k, j) + self.Y(x, k, j + 1))
                drift = rho * Yav * E_ambi * self.m_speciesCharge[k] * self.m_mobility[j, k]
                self.m_flux[k, j] += drift

            # correction flux
            sum_flux = 0.0
            for k in range(self.m_nsp):
                sum_flux -= self.m_flux[k, j]

            sum_ion = 0.0
            for k in self.m_kCharge:
                sum_ion += self.Y(x, k, j)

            # The portion of correction for ions is taken off
            denom = 1.0 - sum_ion
            for k in self.m_kNeutral:
                self.m_flux[k, j] += self.Y(x, k, j) / denom * sum_flux

    def evalElectricField(self, x: np.ndarray, rsd: np.ndarray, diag: np.ndarray,
                           rdt: float, jmin: int, jmax: int) -> None:
        super().evalElectricField(x, rsd, diag, rdt, jmin, jmax)
        if not self.m_do_electric_field:
            return

        if jmin == 0:
            rsd[self.index(c_offset_E, jmin)] = self.E(x, jmin)

        if jmax == self.m_points - 1:
            rsd[self.index(c_offset_E, jmax)] = self.dEdz(x, jmax) - self.rho_e(x, jmax) / epsilon_0

        j0 = max(jmin, 1)
        j1 = min(jmax, self.m_points - 2)
        for j in range(j0, j1 + 1):
            rsd[self.index(c_offset_E, j)] = self.dEdz(x, j) - self.rho_e(x, j) / epsilon_0
            diag[self.index(c_offset_E, j)] = 0

    def evalSpecies(self, x: np.ndarray, rsd: np.ndarray, diag: np.ndarray,
                    rdt: float, jmin: int, jmax: int) -> None:
        super().evalSpecies(x, rsd, diag, rdt, jmin, jmax)
        if not self.m_do_electric_field:
            return

        if jmin == 0:
            for k in self.m_kCharge:
                rsd[self.index(c_offset_Y + k, jmin)] = self.Y(x, k, jmin) - self.Y(x, k, jmin + 1)


class DomainFactory:
    """Factory for creating 1D domain objects.

    Direct reconstruction of Cantera's C++ DomainFactory class.
    """
    _instance: Optional["DomainFactory"] = None
    _mutex = threading.Lock()

    def __init__(self):
        self._creators: dict[str, Callable[[Solution, str], Domain1D]] = {}
        self._register_default_creators()

    def _register_default_creators(self):
        self.reg("inlet", lambda sol, id: Inlet1D(sol, id=id))
        self.reg("empty", lambda sol, id: Empty1D(sol, id=id))
        self.reg("symmetry-plane", lambda sol, id: Symm1D(sol, id=id))
        self.reg("outlet", lambda sol, id: Outlet1D(sol, id=id))
        self.reg("outlet-reservoir", lambda sol, id: OutletRes1D(sol, id=id))
        self.reg("surface", lambda sol, id: Surf1D(sol, id=id))
        self.reg("reacting-surface", lambda sol, id: ReactingSurf1D(sol, id=id))
        self.reg("gas-flow", lambda sol, id: Flow1D(sol, id=id))
        self.reg("ion-flow", lambda sol, id: IonFlow(sol, id=id))
        self.reg("free-flow", self._create_free_flow)
        self.reg("axisymmetric-flow", self._create_axisymmetric_flow)
        self.reg("unstrained-flow", self._create_unstrained_flow)

    def _create_free_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setFreeFlow()
        return flow

    def _create_axisymmetric_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setAxisymmetricFlow()
        return flow

    def _create_unstrained_flow(self, sol: Solution, id: str) -> Flow1D:
        model = getattr(sol.transport, "transportModel", getattr(sol.transport, "transport_model", ""))
        flow = IonFlow(sol, id=id) if model == "ionized-gas" else Flow1D(sol, id=id)
        flow.setUnstrainedFlow()
        return flow

    def reg(self, domain_type: str, creator: Callable[[Solution, str], Domain1D]):
        self._creators[domain_type] = creator

    def create(self, domain_type: str, solution: Solution, id: str = "") -> Domain1D:
        if not id:
            id = domain_type
        if domain_type not in self._creators:
            raise CanteraError("DomainFactory::create", f"Unknown domain type '{domain_type}'")
        return self._creators[domain_type](solution, id)

    @classmethod
    def factory(cls) -> "DomainFactory":
        with cls._mutex:
            if cls._instance is None:
                cls._instance = DomainFactory()
            return cls._instance

    @classmethod
    def deleteFactory(cls) -> None:
        with cls._mutex:
            cls._instance = None


def newDomain(domainType: str, solution: Solution, id: str = "") -> Domain1D:
    """Create a Domain1D object of the specified type."""
    return DomainFactory.factory().create(domainType, solution, id)


def newBoundary1D(domainType: str, solution: Solution, id: str = "") -> Boundary1D:
    """Create a Boundary1D object of the specified type."""
    dom = newDomain(domainType, solution, id)
    if not isinstance(dom, Boundary1D):
        raise CanteraError("newBoundary1D", f"Invalid cast: unable to access '{type(dom).__name__}' as 'Boundary1D'.")
    return dom


def newFlow1D(domainType: str, solution: Solution, id: str = "") -> Flow1D:
    """Create a Flow1D object of the specified type."""
    dom = newDomain(domainType, solution, id)
    if not isinstance(dom, Flow1D):
        raise CanteraError("newFlow1D", f"Invalid cast: unable to access '{type(dom).__name__}' as 'Flow1D'.")
    return dom


'''

content = content[:start_idx] + ionflow_and_factory_code + content[end_idx:]

with open("cantera_python/oned.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully patched oned.py with IonFlow and DomainFactory")
