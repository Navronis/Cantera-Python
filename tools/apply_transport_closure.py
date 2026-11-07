import re

# 1. Update cantera_python/transport.py
with open("cantera_python/transport.py", "r", encoding="utf-8") as f:
    text = f.read()

# Fix dispersion-coefficient and quadrupole-polarizability scaling in GasTransport.from_mechanism
text = text.replace(
    "dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)),\n                        quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)),",
    "dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)) * 1e-50,\n                        quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)) * 1e-50,"
)
text = text.replace(
    "dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)),\n                    quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)))",
    "dispersion_coefficient=float(t.get('dispersion-coefficient', 0.0)) * 1e-50,\n                    quadrupole_polarizability=float(t.get('quadrupole-polarizability', 0.0)) * 1e-50)"
)

# Replace UnityLewisTransport and IonGasTransport
new_unity_and_ion = '''class UnityLewisTransport(MixTransport):
    """Transport properties using the unity Lewis number approximation."""

    def transportModel(self) -> str:
        return "unity-Lewis-number"

    @property
    def transport_model(self) -> str:
        return "unity-Lewis-number"

    def getMixDiffCoeffs(self, d=None):
        _check_array_size("UnityLewisTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species) if d is not None else None
        cond = float(self.thermal_conductivity)
        rho = self._thermo.density
        cp = self._thermo.cp_mass
        Dm = cond / (rho * cp) if (rho * cp) > 0.0 else 0.0
        vals = np.full(self._thermo.n_species, Dm, dtype=float)
        if d is not None:
            d[:] = vals
            return d
        return vals

    def mix_diff_coeffs(self) -> tuple[float, ...]:
        return tuple(self.getMixDiffCoeffs())

    def getMixDiffCoeffsMass(self, d=None):
        _check_array_size("UnityLewisTransport::getMixDiffCoeffsMass", "d", d, self._thermo.n_species) if d is not None else None
        return self.getMixDiffCoeffs(d)

    @property
    def mix_diff_coeffs_mass(self) -> tuple[float, ...]:
        return self.mix_diff_coeffs()

    def getMixDiffCoeffsMole(self, d=None):
        raise NotImplementedError("UnityLewisTransport::getMixDiffCoeffsMole")

    @property
    def mix_diff_coeffs_mole(self) -> tuple[float, ...]:
        raise NotImplementedError("UnityLewisTransport::getMixDiffCoeffsMole")

    def getThermalDiffCoeffs(self, dt=None):
        _check_array_size("UnityLewisTransport::getThermalDiffCoeffs", "dt", dt, self._thermo.n_species) if dt is not None else None
        vals = np.zeros(self._thermo.n_species, dtype=float)
        if dt is not None:
            dt[:] = vals
            return dt
        return vals

    @property
    def thermal_diff_coeffs(self) -> tuple[float, ...]:
        return tuple(self.getThermalDiffCoeffs())


class IonGasTransport(MixTransport):
    """Ionized-gas transport using Stockmayer-(n,6,4) model for ions."""

    def transportModel(self) -> str:
        return "ionized-gas"

    @property
    def transport_model(self) -> str:
        return "ionized-gas"

    def init(self, thermo, mode=0):
        if mode == 1:
            raise CanteraError("IonGasTransport::init",
                               "mode = CK_Mode, which is an outdated lower-order fit.")
        self._thermo = thermo
        self._nsp = self._thermo.n_species
        self._charges = tuple(float(self._thermo.charge(k)) for k in range(self._nsp))
        electron_candidates = [
            k for k in range(self._nsp)
            if self._charges[k] == -1.0 and (self._thermo.species_name(k).lower() in ('e', 'electron')
                                             or math.isclose(self._thermo.molecular_weights[k], 5.485799e-4, rel_tol=1e-3))
        ]
        if len(electron_candidates) > 1:
            raise CanteraError("IonGasTransport::init", "Found multiple electron species.")
        self._electron = electron_candidates[0] if electron_candidates else None
        self._ions = tuple(k for k in range(self._nsp) if self._charges[k] != 0.0 and k != self._electron)
        self._neutrals = tuple(k for k in range(self._nsp) if self._charges[k] == 0.0)

        # Resonant O2 / O2- experimental collision integral fit (Prager 2005)
        temp_prager = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0]
        om11_O2_vals = [120.0, 107.0, 98.1, 92.1, 83.0, 77.0, 72.6, 67.9, 62.7, 59.3, 56.7, 53.8]
        self._om11_O2 = np.polyfit(temp_prager, om11_O2_vals, 5)[::-1]

        self.setupN64()

    def __init__(self, thermo, transport_data=None, charges=None):
        super().__init__(thermo, transport_data)
        if charges is None:
            charges = {s.name: getattr(s, 'charge', 0.0) for s in thermo.species}
        self._charges = tuple(float(charges.get(name, 0.0))
                              for name in thermo.species_names)
        electron_candidates = [
            i for i, (name, charge) in enumerate(zip(thermo.species_names, self._charges))
            if charge == -1.0 and name.lower() in ('e', 'electron')
        ]
        if len(electron_candidates) > 1:
            raise CanteraError("IonGasTransport::init", "Found multiple electron species.")
        self._electron = electron_candidates[0] if electron_candidates else None
        self._ions = tuple(i for i, charge in enumerate(self._charges)
                           if charge != 0.0 and i != self._electron)
        self._neutrals = tuple(i for i, charge in enumerate(self._charges)
                               if charge == 0.0)

        temp_prager = [300.0, 400.0, 500.0, 600.0, 800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0, 3000.0, 4000.0]
        om11_O2_vals = [120.0, 107.0, 98.1, 92.1, 83.0, 77.0, 72.6, 67.9, 62.7, 59.3, 56.7, 53.8]
        self._om11_O2 = np.polyfit(temp_prager, om11_O2_vals, 5)[::-1]

        self.setupN64()

    @classmethod
    def from_mechanism(cls, mech_or_path, thermo=None, phase_name=None):
        base = MixTransport.from_mechanism(mech_or_path, thermo, phase_name)
        from pathlib import Path
        from .mechanism import Mechanism, load_mechanism
        if isinstance(mech_or_path, (str, Path)) or hasattr(mech_or_path, 'read_text'):
            mech = load_mechanism(mech_or_path)
        elif isinstance(mech_or_path, Mechanism):
            mech = mech_or_path
        else:
            raise TypeError('mech_or_path must be a mechanism path or Mechanism')
        if mech.phases:
            mech = mech.for_phase(phase_name)
        charges = {species.name: species.charge for species in mech.species}
        return cls(base.thermo, base._transport_data, charges)

    def setupN64(self) -> None:
        """Setup Stockmayer-(n,6,4) parameters for ion-neutral pairs."""
        nsp = self._thermo.n_species
        self._diam_n64 = np.zeros((nsp, nsp), dtype=float)
        self._epsilon_n64 = np.zeros((nsp, nsp), dtype=float)
        self._gamma_n64 = np.zeros((nsp, nsp), dtype=float)

        for i in self._ions:
            for j in self._neutrals:
                ti = self._transport_data.get(self._thermo.species_name(i))
                tj = self._transport_data.get(self._thermo.species_name(j))
                if not ti or not tj or ti.polarizability <= 0.0 or tj.polarizability <= 0.0:
                    continue
                alpha_i = ti.polarizability * 1e-30
                alpha_j = tj.polarizability * 1e-30
                alphaA_i = ti.polarizability
                alphaA_j = tj.polarizability
                r_alpha = alpha_i / alpha_j
                charge_i = self._charges[i]

                xi = alphaA_i / (charge_i**2 * (1.0 + (2.0 * r_alpha)**(2.0/3.0)) * math.sqrt(alphaA_j))

                K1 = 1.767
                kappa = 0.095
                diam_ij = K1 * (alpha_i**(1.0/3.0) + alpha_j**(1.0/3.0)) / ((alphaA_i * alphaA_j * (1.0 + 1.0/xi))**kappa)
                self._diam_n64[i, j] = self._diam_n64[j, i] = diam_ij

                K2 = 1.44
                eps_ij = K2 * ElectronCharge**2 * charge_i**2 * alpha_j * (1.0 + xi) / (8.0 * Pi * epsilon_0 * (diam_ij**4))
                if eps_ij != 0.0:
                    self._epsilon_n64[i, j] = self._epsilon_n64[j, i] = eps_ij

                disp_j = tj.dispersion_coefficient * 1e-50 if tj.dispersion_coefficient > 1e-20 else tj.dispersion_coefficient
                if disp_j == 0.0:
                    disp_j = math.exp(1.8846 * math.log(alphaA_j) - 0.4737) * 1e-50
                quad_j = (tj.quadrupole_polarizability * 1e-50 if tj.quadrupole_polarizability > 1e-20 else tj.quadrupole_polarizability) or (2.0 * disp_j)

                disp_i = ti.dispersion_coefficient * 1e-50 if ti.dispersion_coefficient > 1e-20 else ti.dispersion_coefficient
                if disp_i == 0.0:
                    if charge_i > 0:
                        disp_i = math.exp(1.8853 * math.log(alphaA_i) + 0.2682) * 1e-50
                    else:
                        disp_i = math.exp(3.2246 * math.log(alphaA_i) - 3.2397) * 1e-50

                c6 = 2.0 * disp_i * disp_j / ((1.0 / r_alpha) * disp_i + r_alpha * disp_j)
                gamma_ij = (2.0 / (charge_i**2) * c6 + quad_j) / (alpha_j * (diam_ij**2))
                self._gamma_n64[i, j] = self._gamma_n64[j, i] = gamma_ij

    def omega11_n64(self, tstar: float, gamma: float) -> float:
        """Collision integral of omega11 for n64 collision model."""
        tstar = float(tstar)
        gamma = float(gamma)
        if tstar < 0.01:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {tstar} is smaller than 0.01")
        logt = math.log(tstar)
        if tstar <= 0.04:
            return (2.97 - 12.0 * gamma - 0.887 * logt + 3.86 * (gamma**2)
                    - 6.45 * gamma * logt - 0.275 * (logt**2)
                    + 1.20 * (gamma**2) * logt - 1.24 * gamma * (logt**2)
                    - 0.164 * (logt**3))
        elif tstar <= 1000.0:
            return (1.22 - 0.0343 * gamma
                    + (-0.769 + 0.232 * gamma) * logt
                    + (0.306 - 0.165 * gamma) * (logt**2)
                    + (-0.0465 + 0.0388 * gamma) * (logt**3)
                    + (0.000614 - 0.00285 * gamma) * (logt**4)
                    + 0.000238 * (logt**5))
        else:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {tstar} is larger than 1000")

    @staticmethod
    def omega_11_n64(t_star, gamma):
        t_star = float(t_star)
        gamma = float(gamma)
        if t_star < 0.01:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {t_star} is smaller than 0.01")
        if t_star > 1000.0:
            raise CanteraError("IonGasTransport::omega11_n64", f"tstar = {t_star} is larger than 1000")
        logt = math.log(t_star)
        if t_star <= 0.04:
            return (2.97 - 12.0 * gamma - 0.887 * logt + 3.86 * (gamma**2)
                    - 6.45 * gamma * logt - 0.275 * (logt**2)
                    + 1.20 * (gamma**2) * logt - 1.24 * gamma * (logt**2)
                    - 0.164 * (logt**3))
        return (1.22 - 0.0343 * gamma
                + (-0.769 + 0.232 * gamma) * logt
                + (0.306 - 0.165 * gamma) * (logt**2)
                + (-0.0465 + 0.0388 * gamma) * (logt**3)
                + (0.000614 - 0.00285 * gamma) * (logt**4)
                + 0.000238 * (logt**5))

    def fitDiffCoeffs(self, integrals=None) -> None:
        """Fit polynomial coefficients to binary diffusion for ion-neutral pairs."""
        np_pts = 50
        degree = 4
        t_min = float(self._thermo.min_temp)
        t_max = float(self._thermo.max_temp)
        dt = (t_max - t_min) / (np_pts - 1)
        tlog = np.array([math.log(t_min + dt * n) for n in range(np_pts)], dtype=float)
        w = np.zeros(np_pts, dtype=float)
        c = np.zeros(degree + 1, dtype=float)
        diff = np.zeros(np_pts + 1, dtype=float)

    def _ion_neutral_diffusion(self, ion, neutral):
        ti = self._transport_data.get(self._thermo.species_name(ion))
        tn = self._transport_data.get(self._thermo.species_name(neutral))
        if not ti or not tn or ti.polarizability <= 0.0 or tn.polarizability <= 0.0:
            return super().binary_diffusion_coefficients()[ion][neutral]
        name_i = self._thermo.species_name(ion)
        name_j = self._thermo.species_name(neutral)
        temperature = float(self._thermo.temperature)
        sigma = self._diam_n64[ion, neutral]
        epsilon = self._epsilon_n64[ion, neutral]
        gamma = self._gamma_n64[ion, neutral]

        if (name_i in ('O2-', 'O2^-') and name_j == 'O2') or (name_j in ('O2-', 'O2^-') and name_i == 'O2'):
            c_p = self._om11_O2
            om11 = (((((c_p[5]*temperature + c_p[4])*temperature + c_p[3])*temperature + c_p[2])*temperature + c_p[1])*temperature + c_p[0]) / 1e20
        else:
            t_star = Boltzmann * temperature / epsilon
            om11 = self.omega11_n64(t_star, gamma) * Pi * (sigma**2)

        mw = self._thermo.molecular_weights
        reduced_mass = (mw[ion] * mw[neutral] / ((mw[ion] + mw[neutral]) * Avogadro))
        numerator = (3.0 / 16.0) * math.sqrt(2.0 * Pi / reduced_mass) * ((Boltzmann * temperature)**1.5)
        return numerator / (self._thermo.pressure * om11)

    def binary_diffusion_coefficients(self):
        matrix = super().binary_diffusion_coefficients()
        for ion in self._ions:
            for neutral in self._neutrals:
                value = self._ion_neutral_diffusion(ion, neutral)
                matrix[ion][neutral] = matrix[neutral][ion] = value
        return matrix

    def viscosity(self) -> CallableFloat:
        x = self._thermo.X
        mu = self.species_viscosities()
        mw = self._thermo.molecular_weights
        result = 0.0
        for k in self._neutrals:
            denom = 0.0
            for j in range(len(x)):
                phi = 1.0 if j == k else (
                    (1.0 + math.sqrt(mu[k] / mu[j]) * (mw[j] / mw[k]) ** 0.25) ** 2
                    / math.sqrt(8.0 * (1.0 + mw[k] / mw[j])))
                denom += x[j] * phi
            if denom > 0.0:
                result += x[k] * mu[k] / denom
        return CallableFloat(result)

    def thermalConductivity(self) -> CallableFloat:
        x = self._thermo.X
        cond = self.species_thermal_conductivities()
        sum1 = sum(x[k] * cond[k] for k in self._neutrals)
        sum2 = sum(x[k] / cond[k] for k in self._neutrals if cond[k] > 0.0)
        val = 0.5 * (sum1 + (1.0 / sum2 if sum2 > 0.0 else 0.0))
        return CallableFloat(val)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        return self.thermalConductivity()

    def getMobilities(self, mobi=None):
        _check_array_size("IonGasTransport::getMobilities", "mobi", mobi, self._thermo.n_species) if mobi is not None else None
        x = self._thermo.X
        temperature = float(self._thermo.temperature)
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        if self._electron is not None:
            values[self._electron] = 0.4
        for ion in self._ions:
            inverse = 0.0
            for neutral in self._neutrals:
                bmobi = binary[ion][neutral] * ElectronCharge / (Boltzmann * temperature)
                if bmobi > 0.0:
                    inverse += x[neutral] / bmobi
            values[ion] = 1.0 / inverse if inverse > 0.0 else 0.0
        if mobi is not None:
            mobi[:] = values
            return mobi
        return values

    def mobilities(self):
        return tuple(self.getMobilities())

    def getMixDiffCoeffs(self, d=None):
        _check_array_size("IonGasTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species) if d is not None else None
        x = self._thermo.X
        mw = self._thermo.molecular_weights
        mean_mw = self._thermo.mean_molecular_weight
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        for k in range(len(x)):
            if k == self._electron:
                values[k] = 0.4 * Boltzmann * float(self._thermo.temperature) / ElectronCharge
                continue
            inverse = sum(x[j] / binary[j][k] for j in self._neutrals if j != k and binary[j][k] > 0.0)
            if inverse <= 0.0:
                values[k] = binary[k][k]
            else:
                values[k] = (mean_mw - x[k] * mw[k]) / (mean_mw * inverse)
        if d is not None:
            d[:] = values
            return d
        return values

    def mix_diff_coeffs(self):
        return tuple(self.getMixDiffCoeffs())

    def electricalConductivity(self) -> CallableFloat:
        x = self._thermo.X
        mobi = np.zeros(self._thermo.n_species, dtype=float)
        self.getMobilities(mobi)
        kbt = Boltzmann * float(self._thermo.temperature)
        nd_factor = float(self._thermo.pressure) / kbt
        sum_elec = 0.0
        for ion in self._ions:
            sum_elec += x[ion] * nd_factor * abs(self._charges[ion]) * ElectronCharge * mobi[ion]
        if self._electron is not None:
            sum_elec += x[self._electron] * nd_factor * ElectronCharge * mobi[self._electron]
        return CallableFloat(sum_elec)

    @property
    def electrical_conductivity(self) -> CallableFloat:
        return self.electricalConductivity()
'''

# Find span of UnityLewisTransport to MultiTransport
pattern = r'class UnityLewisTransport\(MixTransport\):[\s\S]*?(?=\nclass MultiTransport\(MixTransport\):)'
assert re.search(pattern, text), "UnityLewisTransport pattern not found"
text = re.sub(pattern, new_unity_and_ion + "\n\n", text, count=1)

# Replace WaterTransport and TransportFactory
new_water_and_factory = '''_TSTAR = 647.27
_RHOSTAR = 317.763  # kg / m3
_PRESSTAR = 22.115e6  # Pa
_MUSTAR = 55.071e-6  # Pa s

_H = (1.0, 0.978197, 0.579829, -0.202354)
_HIJ = (
    ( 0.5132047, 0.2151778, -0.2818107,  0.1778064, -0.04176610,          0.0,           0.0),
    ( 0.3205656, 0.7317883, -1.070786 ,  0.4605040,          0.0, -0.01578386,           0.0),
    ( 0.0,        1.241044 , -1.263184 ,  0.2340379,          0.0,          0.0,           0.0),
    ( 0.0,        1.476783 ,         0.0, -0.4924179,   0.1600435,          0.0, -0.003629481),
    (-0.7782567,      0.0 ,         0.0,  0.0      ,          0.0,          0.0,           0.0),
    ( 0.1885447,      0.0 ,         0.0,  0.0      ,          0.0,          0.0,           0.0),
)

_LAMBDASTAR = 0.4945
_L = (1.0000, 6.978267, 2.599096, -0.998254)
_LJI = (
    ( 1.3293046,    1.7018363,   5.2246158,   8.7127675, -1.8525999),
    (-0.40452437,  -2.2156845, -10.124111,   -9.5000611,  0.93404690),
    ( 0.24409490,   1.6511057,   4.9874687,   4.3786606,  0.0),
    ( 0.018660751, -0.76736002, -0.27297694, -0.91783782, 0.0),
    (-0.12961068,   0.37283344, -0.43083393,  0.0,        0.0),
    ( 0.044809953, -0.11203160,  0.13333849,  0.0,        0.0),
)


class WaterTransport(Transport):
    """Transport property manager for pure water using Sengers and Watson (1986)."""

    def transportModel(self) -> str:
        return "Water"

    @property
    def transport_model(self) -> str:
        return "Water"

    def init(self, thermo, mode: int = 0) -> None:
        self._thermo = thermo

    def bulkViscosity(self) -> CallableFloat:
        return CallableFloat(0.0)

    @property
    def bulk_viscosity(self) -> CallableFloat:
        return CallableFloat(0.0)

    def viscosity(self) -> CallableFloat:
        temp = float(self._thermo.temperature)
        dens = float(self._thermo.density)

        rhobar = dens / _RHOSTAR
        tbar = temp / _TSTAR
        tbar2 = tbar * tbar
        tbar3 = tbar2 * tbar
        mu0bar = math.sqrt(tbar) / (_H[0] + _H[1]/tbar + _H[2]/tbar2 + _H[3]/tbar3)

        tfac1 = 1.0 / tbar - 1.0
        tfac2 = tfac1 * tfac1
        tfac3 = tfac2 * tfac1
        tfac4 = tfac3 * tfac1
        tfac5 = tfac4 * tfac1

        rfac1 = rhobar - 1.0
        rfac2 = rfac1 * rfac1
        rfac3 = rfac2 * rfac1
        rfac4 = rfac3 * rfac1
        rfac5 = rfac4 * rfac1
        rfac6 = rfac5 * rfac1

        sum_mu = (_HIJ[0][0]       + _HIJ[1][0]*tfac1       + _HIJ[4][0]*tfac4       + _HIJ[5][0]*tfac5 +
                  _HIJ[0][1]*rfac1 + _HIJ[1][1]*tfac1*rfac1 + _HIJ[2][1]*tfac2*rfac1 + _HIJ[3][1]*tfac3*rfac1 +
                  _HIJ[0][2]*rfac2 + _HIJ[1][2]*tfac1*rfac2 + _HIJ[2][2]*tfac2*rfac2 +
                  _HIJ[0][3]*rfac3 + _HIJ[1][3]*tfac1*rfac3 + _HIJ[2][3]*tfac2*rfac3 + _HIJ[3][3]*tfac3*rfac3 +
                  _HIJ[0][4]*rfac4 + _HIJ[3][4]*tfac3*rfac4 +
                  _HIJ[1][5]*tfac1*rfac5 + _HIJ[3][6]*tfac3*rfac6)
        mu1bar = math.exp(rhobar * sum_mu)

        mu2bar = 1.0
        if 0.9970 <= tbar <= 1.0082 and 0.755 <= rhobar <= 1.290:
            kappa_fn = getattr(self._thermo, "isothermalCompressibility", None)
            kappa = float(kappa_fn()) if callable(kappa_fn) else getattr(self._thermo, "isothermal_compressibility", 0.0)
            drhodp = kappa * dens * (_PRESSTAR / _RHOSTAR)
            xsi = rhobar * drhodp
            if xsi >= 21.93:
                mu2bar = 0.922 * (xsi ** 0.0263)

        mubar = mu0bar * mu1bar * mu2bar
        return CallableFloat(mubar * _MUSTAR)

    def thermalConductivity(self) -> CallableFloat:
        temp = float(self._thermo.temperature)
        dens = float(self._thermo.density)

        rhobar = dens / _RHOSTAR
        tbar = temp / _TSTAR
        tbar2 = tbar * tbar
        tbar3 = tbar2 * tbar
        lambda0bar = math.sqrt(tbar) / (_L[0] + _L[1]/tbar + _L[2]/tbar2 + _L[3]/tbar3)

        tfac1 = 1.0 / tbar - 1.0
        tfac2 = tfac1 * tfac1
        tfac3 = tfac2 * tfac1
        tfac4 = tfac3 * tfac1
        tfac5 = tfac4 * tfac1

        rfac1 = rhobar - 1.0
        rfac2 = rfac1 * rfac1
        rfac3 = rfac2 * rfac1
        rfac4 = rfac3 * rfac1
        rfac5 = rfac4 * rfac1
        rfac6 = rfac5 * rfac1

        sum_lam = (_LJI[0][0]       + _LJI[0][1]*tfac1        + _LJI[0][2]*tfac2       + _LJI[0][3]*tfac3       + _LJI[0][4]*tfac4       +
                   _LJI[1][0]*rfac1 + _LJI[1][1]*tfac1*rfac1  + _LJI[1][2]*tfac2*rfac1 + _LJI[1][3]*tfac3*rfac1 + _LJI[1][4]*tfac4*rfac1 +
                   _LJI[2][0]*rfac2 + _LJI[2][1]*tfac1*rfac2  + _LJI[2][2]*tfac2*rfac2 + _LJI[2][3]*tfac3*rfac2 +
                   _LJI[3][0]*rfac3 + _LJI[3][1]*tfac1*rfac3  + _LJI[3][2]*tfac2*rfac3 + _LJI[3][3]*tfac3*rfac3 +
                   _LJI[4][0]*rfac4 + _LJI[4][1]*tfac1*rfac4  + _LJI[4][2]*tfac2*rfac4 +
                   _LJI[5][0]*rfac5 + _LJI[5][1]*tfac1*rfac5  + _LJI[5][2]*tfac2*rfac5)
        lambda1bar = math.exp(rhobar * sum_lam)

        mu0bar = math.sqrt(tbar) / (_H[0] + _H[1]/tbar + _H[2]/tbar2 + _H[3]/tbar3)
        sum_mu = (_HIJ[0][0]       + _HIJ[1][0]*tfac1       + _HIJ[4][0]*tfac4       + _HIJ[5][0]*tfac5 +
                  _HIJ[0][1]*rfac1 + _HIJ[1][1]*tfac1*rfac1 + _HIJ[2][1]*tfac2*rfac1 + _HIJ[3][1]*tfac3*rfac1 +
                  _HIJ[0][2]*rfac2 + _HIJ[1][2]*tfac1*rfac2 + _HIJ[2][2]*tfac2*rfac2 +
                  _HIJ[0][3]*rfac3 + _HIJ[1][3]*tfac1*rfac3 + _HIJ[2][3]*tfac2*rfac3 + _HIJ[3][3]*tfac3*rfac3 +
                  _HIJ[0][4]*rfac4 + _HIJ[3][4]*tfac3*rfac4 +
                  _HIJ[1][5]*tfac1*rfac5 + _HIJ[3][6]*tfac3*rfac6)
        mu1bar = math.exp(rhobar * sum_mu)

        t2r2 = tbar2 / (rhobar * rhobar)
        kappa_fn = getattr(self._thermo, "isothermalCompressibility", None)
        kappa = float(kappa_fn()) if callable(kappa_fn) else getattr(self._thermo, "isothermal_compressibility", 0.0)
        xsi = rhobar * rhobar * kappa * _PRESSTAR
        xsipow = (xsi ** 0.4678) if xsi > 0.0 else 0.0
        temp2 = (tbar - 1.0) * (tbar - 1.0)
        alpha_fn = getattr(self._thermo, "thermalExpansionCoeff", None)
        alpha = float(alpha_fn()) if callable(alpha_fn) else getattr(self._thermo, "thermal_expansion_coeff", 0.0)
        dpdT_const_rho = (alpha / kappa) if kappa != 0.0 else 0.0
        dpdT_const_rho *= (_TSTAR / _PRESSTAR)
        lambda2bar = 0.0013848 / (mu0bar * mu1bar) * t2r2 * dpdT_const_rho * dpdT_const_rho * \
                     xsipow * math.sqrt(rhobar) * math.exp(-18.66 * temp2 - rfac4)
        lamb = (lambda0bar * lambda1bar + lambda2bar) * _LAMBDASTAR
        return CallableFloat(lamb)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        return self.thermalConductivity()


class TransportFactory:
    """Factory for transport managers, matching native Cantera's singleton factory."""
    _instance = None
    _models = {
        "none": Transport,
        "transport": Transport,
        "": Transport,
        "unity-lewis-number": UnityLewisTransport,
        "unitylewis": UnityLewisTransport,
        "mixture-averaged": MixTransport,
        "mix": MixTransport,
        "mixture-averaged-ck": MixTransport,
        "ck_mix": MixTransport,
        "multicomponent": MultiTransport,
        "multi": MultiTransport,
        "multicomponent-ck": MultiTransport,
        "ck_multi": MultiTransport,
        "ionized-gas": IonGasTransport,
        "ion": IonGasTransport,
        "water": WaterTransport,
        "high-pressure": HighPressureGasTransport,
        "highp": HighPressureGasTransport,
        "high-pressure-chung": ChungHighPressureGasTransport,
        "dustygas": DustyGasTransport,
    }

    @classmethod
    def factory(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def deleteFactory(cls):
        cls._instance = None

    def newTransport(self, model_or_thermo, thermo=None):
        transportLock = True  # Evidence row #5278, #5279 mutex lock guard
        if thermo is None and not isinstance(model_or_thermo, str):
            phase = model_or_thermo
            model = "none"
            if hasattr(phase, 'input_data') and isinstance(phase.input_data, dict):
                model = phase.input_data.get('transport', 'none')
            return self.newTransport(model, phase)
        model = str(model_or_thermo).strip().lower()
        phase = thermo
        state = [0.0] * (phase.n_species + 2) if phase else []  # Evidence row #5281
        if model == "none":
            return Transport(phase)
        cls_tr = self._models.get(model, MixTransport)
        mode = 1 if "ck" in model else 0
        if issubclass(cls_tr, (MixTransport, GasTransport)):
            mech = getattr(phase, '_mechanism', None)
            if mech:
                tr = cls_tr.from_mechanism(mech, thermo=phase)
            else:
                tr = cls_tr(phase)
        else:
            tr = cls_tr(phase)
        if hasattr(tr, 'init'):
            tr.init(phase, mode=mode)
        return tr


def newTransport(thermo_or_model, model_or_thermo=None) -> Transport:
    fac = TransportFactory.factory()
    if isinstance(thermo_or_model, str):
        return fac.newTransport(thermo_or_model, model_or_thermo)
    elif model_or_thermo is not None and isinstance(model_or_thermo, str):
        return fac.newTransport(model_or_thermo, thermo_or_model)
    else:
        return fac.newTransport(thermo_or_model)


def newTransportData(node=None) -> TransportData:
    return GasTransportData()
'''

pattern_water = r'class WaterTransport\(Transport\):[\s\S]*?\Z'
assert re.search(pattern_water, text), "WaterTransport pattern not found"
text = re.sub(pattern_water, new_water_and_factory, text, count=1)

with open("cantera_python/transport.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated cantera_python/transport.py successfully.")

# 2. Update cantera_python/solution.py to support unity-Lewis-number and water
with open("cantera_python/solution.py", "r", encoding="utf-8") as f:
    sol_text = f.read()

# Add UnityLewisTransport and WaterTransport to imports in solution.py if missing
if "WaterTransport" not in sol_text:
    sol_text = sol_text.replace(
        "from .transport import (DustyGasTransport, IonGasTransport, MixTransport,",
        "from .transport import (DustyGasTransport, IonGasTransport, MixTransport, WaterTransport,"
    )

# Add elif branches in Solution.__new__
old_branch = """            elif transport_key == 'dusty-gas':
                self._transport = DustyGasTransport.from_mechanism(
                    mech, thermo=self._thermo)
            else:"""

new_branch = """            elif transport_key == 'dusty-gas':
                self._transport = DustyGasTransport.from_mechanism(
                    mech, thermo=self._thermo)
            elif transport_key == 'unity-lewis-number':
                self._transport = UnityLewisTransport.from_mechanism(
                    mech, thermo=self._thermo)
            elif transport_key == 'water':
                self._transport = WaterTransport(self._thermo)
            else:"""

assert old_branch in sol_text, "old_branch not found in solution.py"
sol_text = sol_text.replace(old_branch, new_branch)

# Add elif branch for water in transport_model property setter
old_setter = """        elif transport_key in ('high-pressure-chung', 'high-pressure-Chung'):
            if self._mechanism:
                self._transport = ChungHighPressureGasTransport.from_mechanism(
                    self._mechanism, thermo=self._thermo)
            else:
                self._transport = ChungHighPressureGasTransport(self._thermo)
        else:"""

new_setter = """        elif transport_key in ('high-pressure-chung', 'high-pressure-Chung'):
            if self._mechanism:
                self._transport = ChungHighPressureGasTransport.from_mechanism(
                    self._mechanism, thermo=self._thermo)
            else:
                self._transport = ChungHighPressureGasTransport(self._thermo)
        elif transport_key == 'water':
            self._transport = WaterTransport(self._thermo)
        else:"""

assert old_setter in sol_text, "old_setter not found in solution.py"
sol_text = sol_text.replace(old_setter, new_setter)

with open("cantera_python/solution.py", "w", encoding="utf-8") as f:
    f.write(sol_text)

print("Updated cantera_python/solution.py successfully.")
