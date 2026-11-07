"""Patch cantera_python/transport.py and solution.py with complete HighPressureGasTransport implementations."""

from pathlib import Path

HP_CLASSES_CODE = '''class CallableFloat(float):
    """A float subclass that is callable as a zero-argument function."""
    def __call__(self):
        return float(self)


class HighPressureGasTransportBase(MixTransport):
    """Base class for dense gas / high-pressure transport models."""

    def __init__(self, thermo, transport_data: Mapping[str, GasTransportData] | None = None):
        if not hasattr(thermo, "setSpeciesCoeffs") and hasattr(thermo, "thermo"):
            thermo = thermo.thermo
        super().__init__(thermo, transport_data)
        nsp = self._thermo.n_species
        if not hasattr(self._thermo, "setSpeciesCoeffs") and not hasattr(self._thermo, "critTemperature"):
            raise ValueError(
                "High-pressure gas transport requires a phase with species critical properties")
        self.initializeCriticalProperties()
        self._P_corr_ij = np.ones((nsp, nsp), dtype=float)
        self._takahashi_factors = self._P_corr_ij
        self.getTransportData()

    def getTransportData(self) -> dict:
        """Load and cache species critical and transport data (e.g. acentric factors)."""
        nsp = self._thermo.n_species
        self._w_ac = np.zeros(nsp, dtype=float)
        if hasattr(self._thermo, "_acentric"):
            self._w_ac = np.array(self._thermo._acentric, dtype=float).copy()

        crit_db = None
        for k in range(nsp):
            if self._w_ac[k] == 0.0:
                sp = self._thermo.species(k) if hasattr(self._thermo, "species") else None
                if sp and hasattr(sp, "acentric_factor") and sp.acentric_factor:
                    self._w_ac[k] = float(sp.acentric_factor)
                else:
                    if crit_db is None:
                        crit_db = self._loadCriticalPropertiesDb()
                    sp_name = self._thermo.species_name(k).upper()
                    if crit_db and sp_name in crit_db:
                        params = crit_db[sp_name].get("critical-parameters", {})
                        if "acentric-factor" in params:
                            self._w_ac[k] = float(params["acentric-factor"])
        return {"acentric_factors": self._w_ac}

    @staticmethod
    def _loadCriticalPropertiesDb() -> dict:
        import yaml
        from pathlib import Path
        for cand in [
            Path(__file__).resolve().parent.parent.parent / "cantera-original" / "data" / "critical-properties.yaml",
            Path("data/critical-properties.yaml"),
        ]:
            if cand.exists():
                try:
                    data = yaml.safe_load(cand.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "species" in data:
                        return {s["name"].upper(): s for s in data["species"] if "name" in s}
                except Exception:
                    pass
        return {}

    def initializeCriticalProperties(self) -> None:
        """Extract and cache pure-species critical properties (Tc, Pc, Vc, Zc)."""
        nsp = self._thermo.n_species
        self._tcrit = np.zeros(nsp, dtype=float)
        self._pcrit = np.zeros(nsp, dtype=float)
        self._vcrit = np.zeros(nsp, dtype=float)
        self._zcrit = np.zeros(nsp, dtype=float)

        molefracs = np.asarray(self._thermo.X, dtype=float).copy()
        mf_temp = np.zeros(nsp, dtype=float)
        try:
            for i in range(nsp):
                mf_temp[i] = 1.0
                self._thermo.X = mf_temp
                tc = float(self._thermo.critTemperature())
                if tc > 1e4:
                    raise ValueError(
                        f"Species '{self._thermo.species_name(i)}' must have critical "
                        "properties defined or non-zero cubic parameters. Check the "
                        "species definition or the thermo data file.")
                self._tcrit[i] = tc
                self._pcrit[i] = float(self._thermo.critPressure())
                self._vcrit[i] = float(self._thermo.critVolume())
                self._zcrit[i] = float(self._thermo.critCompressibility())
                mf_temp[i] = 0.0
        finally:
            self._thermo.X = molefracs

    def Tcrit_i(self, i: int) -> float:
        return float(self._tcrit[i])

    def Pcrit_i(self, i: int) -> float:
        return float(self._pcrit[i])

    def Vcrit_i(self, i: int) -> float:
        return float(self._vcrit[i])

    def Zcrit_i(self, i: int) -> float:
        return float(self._zcrit[i])

    def takahashiCorrectionFactor(self, i: int, j: int) -> float:
        """Takahashi high-pressure correction factor for binary diffusion pair (i, j)."""
        T = self._thermo.temperature
        P = self._thermo.pressure
        xi = max(Tiny, float(self._thermo.X[i]))
        xj = max(Tiny, float(self._thermo.X[j]))
        sum_x_ij = xi + xj
        xi /= sum_x_ij
        xj /= sum_x_ij
        Tr_ij = T / (xi * self.Tcrit_i(i) + xj * self.Tcrit_i(j))
        Pr_ij = P / (xi * self.Pcrit_i(i) + xj * self.Pcrit_i(j))
        fac = takahashi_correction_factor(Pr_ij, Tr_ij)
        return max(Tiny, fac)

    def updateCorrectionFactors(self) -> None:
        """Update the matrix of Takahashi high-pressure diffusion correction factors."""
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        T = self._thermo.temperature
        P = self._thermo.pressure
        for i in range(nsp):
            for j in range(nsp):
                x_i = max(Tiny, float(molefracs[i]))
                x_j = max(Tiny, float(molefracs[j]))
                sum_x_ij = x_i + x_j
                x_i /= sum_x_ij
                x_j /= sum_x_ij
                Tr_ij = T / (x_i * self.Tcrit_i(i) + x_j * self.Tcrit_i(j))
                Pr_ij = P / (x_i * self.Pcrit_i(i) + x_j * self.Pcrit_i(j))
                P_corr_ij = takahashi_correction_factor(Pr_ij, Tr_ij)
                if P_corr_ij < 0.0:
                    P_corr_ij = Tiny
                self._P_corr_ij[i, j] = P_corr_ij
        self._takahashi_factors = self._P_corr_ij

    def binary_diffusion_coefficients(self) -> np.ndarray:
        """Binary diffusion coefficients with Takahashi high-pressure corrections."""
        bdiff_low = np.asarray(super().binary_diffusion_coefficients(), dtype=float)
        self.updateCorrectionFactors()
        return bdiff_low * self._P_corr_ij

    @property
    def binary_diff_coeffs(self) -> np.ndarray:
        return self.binary_diffusion_coefficients()

    def getBinaryDiffCoeffs(self, ld: int, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getBinaryDiffCoeffs",
                          "binary diffusion coefficients", d, ld * nsp)
        if ld < nsp:
            raise ValueError("HighPressureGasTransport::getBinaryDiffCoeffs: ld is too small")
        bdiff = self.binary_diffusion_coefficients()
        for j in range(nsp):
            for i in range(nsp):
                d[ld * j + i] = bdiff[i, j]

    def mix_diff_coeffs(self) -> np.ndarray:
        """Mixture-averaged diffusion coefficients with high-pressure corrections."""
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (mmw - molefracs[i] * mw[i]) / (mmw * sum2)
        return d

    def getMixDiffCoeffs(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffs",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs()
        d[:nsp] = res

    def mix_diff_coeffs_mole(self) -> np.ndarray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum2 = sum(molefracs[j] / bdiff[j, i] for j in range(nsp) if j != i)
            if sum2 <= 0.0:
                d[i] = bdiff[i, i]
            else:
                d[i] = (1.0 - molefracs[i]) / sum2
        return d

    def getMixDiffCoeffsMole(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffsMole",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs_mole()
        d[:nsp] = res

    def mix_diff_coeffs_mass(self) -> np.ndarray:
        self.updateCorrectionFactors()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        bdiff = np.asarray(super().binary_diffusion_coefficients(), dtype=float) * self._P_corr_ij
        if nsp == 1:
            return np.array([bdiff[0, 0]], dtype=float)
        mmw = self._thermo.mean_molecular_weight
        mw = self._thermo.molecular_weights
        d = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            sum1 = 0.0
            sum2 = 0.0
            for j in range(nsp):
                if j == i:
                    continue
                sum1 += molefracs[j] / bdiff[i, j]
                sum2 += molefracs[j] * mw[j] / bdiff[i, j]
            denom = mmw - mw[i] * molefracs[i]
            sum2 *= molefracs[i] / denom if abs(denom) > Tiny else 0.0
            d[i] = 1.0 / (sum1 + sum2)
        return d

    def getMixDiffCoeffsMass(self, d: np.ndarray) -> None:
        nsp = self._thermo.n_species
        _check_array_size("HighPressureGasTransport::getMixDiffCoeffsMass",
                          "diffusion coefficients", d, nsp)
        res = self.mix_diff_coeffs_mass()
        d[:nsp] = res


class HighPressureGasTransport(HighPressureGasTransportBase):
    """High-pressure gas transport using Lucas viscosity and Ely & Hanley thermal conductivity."""

    m_ref_MW = 16.04
    m_ref_Tc = 190.4
    m_ref_Vc = 0.0986
    m_ref_Zc = 0.288
    m_ref_rhoc = 0.1628
    m_ref_acentric_factor = 0.011

    def transportModel(self) -> str:
        return "HighPressure"

    @property
    def transport_model(self) -> str:
        return "high-pressure"

    def init(self, thermo, mode: int = 0) -> None:
        super().init(thermo, mode)
        self.initializeCriticalProperties()
        self._P_corr_ij = np.ones((self._thermo.n_species, self._thermo.n_species), dtype=float)

    def thetaShapeFactor(self, Tr: float, Vr: float, acentric_factor: float) -> float:
        T_p = min(max(Tr, 0.5), 2.0)
        V_p = min(max(Vr, 0.5), 2.0)
        return 1.0 + (acentric_factor - self.m_ref_acentric_factor) * (
            0.090569 - 0.862762 * math.log(T_p) + (0.316636 - 0.465684 / T_p) * (V_p - 0.5)
        )

    def phiShapeFactor(self, Tr: float, Vr: float, Zc: float, acentric_factor: float) -> float:
        T_p = min(max(Tr, 0.5), 2.0)
        V_p = min(max(Vr, 0.5), 2.0)
        return (1.0 + (acentric_factor - self.m_ref_acentric_factor) * (
            0.394901 * (V_p - 1.023545) - 0.932813 * (V_p - 0.754639) * math.log(T_p)
        )) * (self.m_ref_Zc / Zc)

    def elyHanleyDiluteReferenceViscosity(self, T0: float) -> float:
        correlation_viscosity_conversion = 1e-7
        if T0 > 10000.0:
            T0 = 10000.0
        c = (2.907741307e6, -3.312874033e6, 1.608101838e6,
             -4.331904871e5, 7.062481330e4, -7.116620750e3,
             4.325174400e2, -1.445911210e1, 2.037119479e-1)
        mu_0 = sum(c[i] * (T0 ** ((i + 1.0 - 4.0) / 3.0)) for i in range(9))
        return correlation_viscosity_conversion * mu_0

    def elyHanleyDilutePureSpeciesViscosity(self, V: float, Tc: float, Vc: float,
                                            Zc: float, acentric_factor: float,
                                            mw: float) -> float:
        Tr = self._thermo.temperature / Tc
        Vr = V / Vc
        theta_i = self.thetaShapeFactor(Tr, Vr, acentric_factor)
        phi_i = self.phiShapeFactor(Tr, Vr, Zc, acentric_factor)
        f_fac = (Tc / self.m_ref_Tc) * theta_i
        h_fac = (Vc / self.m_ref_Vc) * phi_i
        T_0 = self._temp / f_fac
        mu_0 = self.elyHanleyDiluteReferenceViscosity(T_0)
        F = math.sqrt(f_fac * (mw / self.m_ref_MW)) * (h_fac ** (-2.0 / 3.0))
        return mu_0 * F

    def elyHanleyReferenceThermalConductivity(self, rho0: float, T0: float) -> float:
        correlation_factor = 1e-3
        mu_0 = self.elyHanleyDiluteReferenceViscosity(T0)
        Lambda_ref_star = (15.0 * GasConstant / (4.0 * self.m_ref_MW)) * mu_0
        b = (-2.52762920e-1, 3.34328590e-1, 1.12, 1.680e2)
        Lambda_ref_1 = (b[0] + b[1] * ((b[2] - math.log(T0 / b[3])) ** 2.0)) * rho0
        a = (-7.197708227, 8.5678222640e1, 1.2471834689e1,
             -9.8462522975e2, 3.5946850007e-1, 6.9798412538e1,
             -8.7288332851e2)
        delta_lambda_ref = math.exp(a[0] + a[1] / T0) * (
            math.exp((a[2] + a[3] * (T0 ** -1.5)) * (rho0 ** 0.1)
                     + (rho0 / self.m_ref_rhoc - 1.0) * math.sqrt(rho0) * (
                         a[4] + a[5] / T0 + a[6] * (T0 ** -2.0))) - 1.0
        )
        return Lambda_ref_star + (Lambda_ref_1 + delta_lambda_ref) * correlation_factor

    def computeMixtureParameters(self) -> None:
        Tc_mix = 0.0
        Pc_mix_n = 0.0
        Pc_mix_d = 0.0
        MW_mix = self._thermo.mean_molecular_weight
        FP_mix_o = 0.0
        FQ_mix_o = 0.0
        MW_H = self._mw[0]
        MW_L = self._mw[0]
        tKelvin = self._thermo.temperature
        try:
            P_vap_mix = float(self._thermo.satPressure(tKelvin))
        except Exception:
            P_vap_mix = 0.0

        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        x_H = molefracs[0]

        for i in range(nsp):
            Tc = self.Tcrit_i(i)
            Tr = tKelvin / Tc
            Zc = self.Zcrit_i(i)
            Tc_mix += Tc * molefracs[i]
            Pc_mix_n += molefracs[i] * Zc
            Pc_mix_d += molefracs[i] * self.Vcrit_i(i)

            if self._mw[i] > MW_H:
                MW_H = self._mw[i]
                x_H = molefracs[i]
            elif self._mw[i] < MW_L:
                MW_L = self._mw[i]

            pascals_to_bar = 1e-5
            dipole_ii = float(self._dipole[i])
            mu_ri = 52.46 * (dipole_ii ** 2.0) * (self.Pcrit_i(i) * pascals_to_bar) / (Tc ** 2.0)
            FP_mix_o += molefracs[i] * self.polarityCorrectionFactor(mu_ri, Tr, Zc)

            spnames = self._thermo.species_names
            if spnames[i] == "He":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(1.38, Tr, self._mw[i])
            elif spnames[i] == "H2":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(0.76, Tr, self._mw[i])
            elif spnames[i] == "D2":
                FQ_mix_o += molefracs[i] * self.quantumCorrectionFactor(0.52, Tr, self._mw[i])
            else:
                FQ_mix_o += molefracs[i]

        Tr_mix = tKelvin / Tc_mix
        Pc_mix = GasConstant * Tc_mix * Pc_mix_n / Pc_mix_d
        Pr_mix = self._thermo.pressure / Pc_mix

        ratio = MW_H / MW_L
        A = 1.0
        if ratio > 9.0 and 0.05 < x_H < 0.7:
            A = 1.0 - 0.01 * (ratio ** 0.87)
        FQ_mix_o *= A

        self._FQ_mix_o = FQ_mix_o
        self._FP_mix_o = FP_mix_o
        self._Tc_mix = Tc_mix
        self._Tr_mix = Tr_mix
        self._Pc_mix = Pc_mix
        self._Pr_mix = Pr_mix
        self._MW_mix = MW_mix
        self._P_vap_mix = P_vap_mix

    def lowPressureNondimensionalViscosity(self, Tr: float, FP: float, FQ: float) -> float:
        first_term = 0.807 * (Tr ** 0.618) - 0.357 * math.exp(-0.449 * Tr)
        second_term = 0.340 * math.exp(-4.058 * Tr) + 0.018
        return (first_term + second_term) * FP * FQ

    def highPressureNondimensionalViscosity(self, Tr: float, Pr: float, FP_low: float,
                                           FQ_low: float, P_vap: float, P_crit: float) -> float:
        Z_1 = self.lowPressureNondimensionalViscosity(Tr, FP_low, FQ_low)
        if Tr <= 1.0:
            if Pr < P_vap / P_crit:
                alpha = 3.262 + 14.98 * (Pr ** 5.508)
                beta = 1.390 + 5.746 * Pr
                Z_2 = 0.600 + 0.760 * (Pr ** alpha) + (0.6990 * (Pr ** beta) - 0.60) * (1.0 - Tr)
            else:
                raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                                   f"State is outside the limits of the Lucas model, Pr ({Pr}) >= P_vap / P_crit ({P_vap / P_crit}) when Tr ({Tr}) <= 1.0")
        elif 1.0 < Tr < 40.0:
            if 0.0 < Pr <= 100.0:
                a_1 = 1.245e-3; a_2 = 5.1726; gamma = -0.3286
                a = a_1 * math.exp(a_2 * (Tr ** gamma)) / Tr
                b_1 = 1.6553; b_2 = 1.2723
                b = a * (b_1 * Tr - b_2)
                c_1 = 0.4489; c_2 = 3.0578; delta = -37.7332
                c = c_1 * math.exp(c_2 * (Tr ** delta)) / Tr
                d_1 = 1.7368; d_2 = 2.2310; epsilon = -7.6351
                d = d_1 * math.exp(d_2 * (Tr ** epsilon)) / Tr
                e = 1.3088
                f_1 = 0.9425; f_2 = -0.1853; zeta = 0.4489
                f = f_1 * math.exp(f_2 * (Tr ** zeta))
                Z_2 = Z_1 * (1.0 + (a * (Pr ** e)) / (b * (Pr ** f) + ((1.0 + c * (Pr ** d)) ** -1.0)))
            else:
                raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                                   f"The value of Pr ({Pr}) is outside the limits of the Lucas model, valid values are 0.0 < Pr <= 100")
        else:
            raise CanteraError("HighPressureGasTransport::highPressureNondimensionalViscosity",
                               f"The value of Tr ({Tr}) is outside the limits of the Lucas model, valid values are 1.0 < Tr < 40")

        Y = Z_2 / Z_1
        FP = (1.0 + (FP_low - 1.0) * (Y ** -3.0)) / FP_low
        FQ = (1.0 + (FQ_low - 1.0) * (1.0 / Y - 0.007 * (math.log(Y) ** 4.0))) / FQ_low
        return Z_2 * FP * FQ

    def quantumCorrectionFactor(self, Q: float, Tr: float, MW: float) -> float:
        sgn = 1.0 if (Tr - 12.0) >= 0 else -1.0
        return 1.22 * (Q ** 0.15) * (1.0 + 0.00385 * (((Tr - 12.0) ** 2.0) ** (1.0 / MW)) * sgn)

    def polarityCorrectionFactor(self, mu_r: float, Tr: float, Z_crit: float) -> float:
        if mu_r < 0.022:
            return 1.0
        elif mu_r < 0.075:
            return 1.0 + 30.55 * (max(0.292 - Z_crit, 0.0) ** 1.72)
        else:
            return 1.0 + 30.55 * (max(0.292 - Z_crit, 0.0) ** 1.72) * abs(0.96 + 0.1 * (Tr - 0.7))

    @property
    def viscosity(self) -> CallableFloat:
        """High-pressure gas viscosity including Lucas dense fluid corrections."""
        self.computeMixtureParameters()
        nondimensional_viscosity = self.highPressureNondimensionalViscosity(
            self._Tr_mix, self._Pr_mix, self._FP_mix_o, self._FQ_mix_o,
            self._P_vap_mix, self._Pc_mix)
        numerator = GasConstant * self._Tc_mix * (Avogadro ** 2.0)
        denominator = (self._MW_mix ** 3.0) * (self._Pc_mix ** 4.0)
        xi = (numerator / denominator) ** (1.0 / 6.0)
        return CallableFloat(nondimensional_viscosity / xi)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        """High-pressure mixture thermal conductivity using Ely & Hanley corresponding states."""
        return CallableFloat(self.thermalConductivity())

    def thermalConductivity(self) -> float:
        self.update_T()
        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        cp_0_R = np.zeros(nsp, dtype=float)
        self._thermo.getCp_R_ref(cp_0_R)
        f_int = 1.32

        Lambda_1_i = np.zeros(nsp, dtype=float)
        f_i = np.zeros(nsp, dtype=float)
        h_i = np.zeros(nsp, dtype=float)
        V_k = np.zeros(nsp, dtype=float)
        if hasattr(self._thermo, "getPartialMolarVolumes"):
            self._thermo.getPartialMolarVolumes(V_k)
        elif hasattr(self._thermo, "partial_molar_volumes"):
            V_k[:] = self._thermo.partial_molar_volumes

        for i in range(nsp):
            mu_i = self.elyHanleyDilutePureSpeciesViscosity(
                V_k[i], self.Tcrit_i(i), self.Vcrit_i(i), self.Zcrit_i(i),
                self._w_ac[i], self._mw[i])
            Lambda_1_i[i] = (mu_i / self._mw[i]) * f_int * GasConstant * (cp_0_R[i] - 2.5)
            Tr = self._thermo.temperature / self.Tcrit_i(i)
            Vr = V_k[i] / self.Vcrit_i(i)
            theta_i = self.thetaShapeFactor(Tr, Vr, self._w_ac[i])
            phi_i = self.phiShapeFactor(Tr, Vr, self.Zcrit_i(i), self._w_ac[i])
            f_i[i] = (self.Tcrit_i(i) / self.m_ref_Tc) * theta_i
            h_i[i] = (self.Vcrit_i(i) / self.m_ref_Vc) * phi_i

        h_m = 0.0
        f_m = 0.0
        mw_m = 0.0
        Lambda_1_m = 0.0
        for i in range(nsp):
            for j in range(nsp):
                Lambda_1_ij = 2.0 * Lambda_1_i[i] * Lambda_1_i[j] / (Lambda_1_i[i] + Lambda_1_i[j] + Tiny)
                Lambda_1_m += molefracs[i] * molefracs[j] * Lambda_1_ij
                f_ij = math.sqrt(f_i[i] * f_i[j])
                h_ij = 0.125 * ((h_i[i] ** (1.0 / 3.0) + h_i[j] ** (1.0 / 3.0)) ** 3.0)
                mw_ij_inv = 0.5 * (self._mw[i] + self._mw[j]) / (self._mw[i] * self._mw[j])
                f_m += molefracs[i] * molefracs[j] * f_ij * h_ij
                h_m += molefracs[i] * molefracs[j] * h_ij
                mw_m += molefracs[i] * molefracs[j] * math.sqrt(mw_ij_inv * f_ij) * (h_ij ** (-4.0 / 3.0))

        f_m /= h_m
        mw_m = (mw_m ** -2.0) * f_m * (h_m ** (-8.0 / 3.0))
        kg_m3_to_g_cm3 = 1e-3
        rho_0 = self._thermo.density * h_m * kg_m3_to_g_cm3
        T_0 = self._temp / f_m

        Lambda_2_ref = self.elyHanleyReferenceThermalConductivity(rho_0, T_0)
        F_m = math.sqrt(f_m * self.m_ref_MW / mw_m) * (h_m ** (-2.0 / 3.0))
        Lambda_2_m = F_m * Lambda_2_ref
        return float(Lambda_1_m + Lambda_2_m)


def neufeldCollisionIntegral(T_star: float) -> float:
    """Neufeld collision integral for dimensionless temperature (Poling Eqn 9-4.3)."""
    A = 1.16145
    B = 0.14874
    C = 0.52487
    D = 0.77320
    E = 2.16178
    F = 2.43787
    return A / (T_star ** B) + C / math.exp(D * T_star) + E / math.exp(F * T_star)


class ChungHighPressureGasTransport(HighPressureGasTransportBase):
    """High-pressure gas transport using the Chung et al. dense gas correlation."""

    def transportModel(self) -> str:
        return "high-pressure-Chung"

    @property
    def transport_model(self) -> str:
        return "high-pressure-Chung"

    def init(self, thermo, mode: int = 0) -> None:
        super().init(thermo, mode)
        self.initializeCriticalProperties()
        self.initializePureFluidProperties()
        self._P_corr_ij = np.ones((self._thermo.n_species, self._thermo.n_species), dtype=float)

    def initializePureFluidProperties(self) -> None:
        nsp = self._thermo.n_species
        self.m_sigma_i = np.zeros(nsp, dtype=float)
        self.m_epsilon_over_k_i = np.zeros(nsp, dtype=float)
        self.m_acentric_factor_i = np.zeros(nsp, dtype=float)
        self.m_MW_i = np.zeros(nsp, dtype=float)
        self.m_kappa_i = np.zeros(nsp, dtype=float)
        for i in range(nsp):
            m3_per_kmol_to_cm3_per_mol = 1e3
            Vc = self.Vcrit_i(i) * m3_per_kmol_to_cm3_per_mol
            self.m_sigma_i[i] = 0.809 * (Vc ** (1.0 / 3.0))
            self.m_epsilon_over_k_i[i] = self.Tcrit_i(i) / 1.2593
            self.m_kappa_i[i] = 0.0
            self.m_acentric_factor_i[i] = self._w_ac[i]
            self.m_MW_i[i] = self._mw[i]

    def computeMixtureParameters(self) -> None:
        self.m_sigma_mix = 0.0
        self.m_epsilon_over_k_mix = 0.0
        self.m_MW_mix = 0.0
        self.m_acentric_factor_mix = 0.0
        self.m_mu_mix = 0.0
        self.m_kappa_mix = 0.0
        self.m_Tc_mix = 0.0
        self.m_Vc_mix = 0.0
        self.m_mu_r_mix = 0.0

        nsp = self._thermo.n_species
        molefracs = np.asarray(self._thermo.X, dtype=float)
        for i in range(nsp):
            for j in range(nsp):
                sigma_ij = math.sqrt(self.m_sigma_i[i] * self.m_sigma_i[j])
                self.m_sigma_mix += molefracs[i] * molefracs[j] * (sigma_ij ** 3.0)
                epsilon_over_k_ij = math.sqrt(self.m_epsilon_over_k_i[i] * self.m_epsilon_over_k_i[j])
                self.m_epsilon_over_k_mix += molefracs[i] * molefracs[j] * epsilon_over_k_ij * (sigma_ij ** 3.0)
                MW_ij = (2.0 * self.m_MW_i[i] * self.m_MW_i[j]) / (self.m_MW_i[i] + self.m_MW_i[j])
                self.m_MW_mix += molefracs[i] * molefracs[j] * epsilon_over_k_ij * (sigma_ij ** 2.0) * math.sqrt(MW_ij)
                acentric_factor_ij = 0.5 * (self.m_acentric_factor_i[i] + self.m_acentric_factor_i[j])
                self.m_acentric_factor_mix += molefracs[i] * molefracs[j] * acentric_factor_ij * (sigma_ij ** 3.0)
                dipole_ii = float(self._dipole[i])
                dipole_jj = float(self._dipole[j])
                self.m_mu_mix += molefracs[i] * molefracs[j] * ((dipole_ii * dipole_jj) ** 2.0) / (sigma_ij ** 3.0)
                kappa_ij = math.sqrt(self.m_kappa_i[i] * self.m_kappa_i[j])
                self.m_kappa_mix += molefracs[i] * molefracs[j] * kappa_ij

        self.m_sigma_mix = self.m_sigma_mix ** (1.0 / 3.0)
        self.m_epsilon_over_k_mix /= (self.m_sigma_mix ** 3.0)
        self.m_MW_mix = (self.m_MW_mix / (self.m_epsilon_over_k_mix * (self.m_sigma_mix ** 2.0))) ** 2.0
        self.m_acentric_factor_mix /= (self.m_sigma_mix ** 3.0)
        self.m_mu_mix = ((self.m_sigma_mix ** 3.0) * self.m_mu_mix) ** (1.0 / 4.0)
        self.m_Tc_mix = 1.2593 * self.m_epsilon_over_k_mix
        self.m_Vc_mix = (self.m_sigma_mix / 0.809) ** 3.0
        self.m_mu_r_mix = 131.3 * self.m_mu_mix / math.sqrt(self.m_Vc_mix * self.m_Tc_mix)

    def lowPressureViscosity(self, T: float, T_star: float, MW: float,
                             acentric_factor: float, mu_r: float,
                             sigma: float, kappa: float) -> float:
        omega = neufeldCollisionIntegral(T_star)
        Fc = 1.0 - 0.2756 * acentric_factor + 0.059035 * (mu_r ** 4.0) + kappa
        viscosity = Fc * (26.69 * math.sqrt(MW * T) / (sigma * sigma * omega))
        micropoise_to_pascals_second = 1e-7
        return micropoise_to_pascals_second * viscosity

    def highPressureViscosity(self, T_star: float, MW: float, rho: float,
                              Vc: float, Tc: float, acentric_factor: float,
                              mu_r: float, kappa: float) -> float:
        a = (6.324, 1.210e-3, 5.283, 6.623, 19.745, -1.900, 24.275, 0.7972, -0.2382, 0.06863)
        b = (50.412, -1.154e-3, 254.209, 38.096, 7.630, -12.537, 3.450, 1.117, 0.06770, 0.3479)
        c = (-51.680, -6.257e-3, -168.48, -8.464, -14.354, 4.958, -11.291, 0.01235, -0.8163, 0.5926)
        d = (1189.0, 0.03728, 3898.0, 31.42, 31.53, -18.15, 69.35, -4.117, 4.025, -0.727)
        mu_r4 = mu_r ** 4.0
        E = [a[k] + b[k] * acentric_factor + c[k] * mu_r4 + d[k] * kappa for k in range(10)]
        y = rho * Vc / 6.0
        G_1 = (1.0 - 0.5 * y) / ((1.0 - y) ** 3.0)
        G_2 = (E[0] * ((1.0 - math.exp(-E[3] * y)) / y) + E[1] * G_1 * math.exp(E[4] * y) + E[2] * G_1) / (E[0] * E[3] + E[1] + E[2])
        eta_2 = E[6] * y * y * G_2 * math.exp(E[7] + E[8] / T_star + E[9] / (T_star * T_star))
        omega = neufeldCollisionIntegral(T_star)
        Fc = 1.0 - 0.2756 * acentric_factor + 0.059035 * mu_r4 + kappa
        eta_1 = (math.sqrt(T_star) / omega) * (Fc * (1.0 / G_2 + E[5] * y)) + eta_2
        eta = eta_1 * 36.344 * math.sqrt(MW * Tc) / (Vc ** (2.0 / 3.0))
        return eta

    @property
    def viscosity(self) -> CallableFloat:
        """High-pressure gas viscosity using Chung dense fluid correlation."""
        self.computeMixtureParameters()
        tKelvin = self._thermo.temperature
        T_star = tKelvin / self.m_epsilon_over_k_mix
        kg_per_m3_to_mol_per_cm3 = (1.0 / self.m_MW_mix) * 1e-3
        molar_density = self._thermo.density * kg_per_m3_to_mol_per_cm3
        viscosity = self.highPressureViscosity(
            T_star, self.m_MW_mix, molar_density, self.m_Vc_mix, self.m_Tc_mix,
            self.m_acentric_factor_mix, self.m_mu_r_mix, self.m_kappa_mix)
        micropoise_to_pascals_second = 1e-7
        return CallableFloat(viscosity * micropoise_to_pascals_second)

    def highPressureThermalConductivity(self, T: float, T_star: float, MW: float,
                                        rho: float, Cv: float, Vc: float,
                                        Tc: float, sigma: float,
                                        acentric_factor: float, mu_r: float,
                                        kappa: float) -> float:
        viscosity = self.lowPressureViscosity(T, T_star, MW, acentric_factor, mu_r, sigma, kappa)
        M_prime = MW / 1000.0
        a = (2.44166, -5.0924e-1, 6.6107, 1.4543e1, 7.9274e-1, -5.8634, 9.1089e1)
        b = (7.4824e-1, -1.5094, 5.6207, -8.9139, 8.2019e-1, 1.2801e1, 1.2811e2)
        c = (-9.1858e-1, -4.9991e1, 6.4760e1, -5.6379, -6.9369e-1, 9.5893, -5.4217e1)
        d = (1.2172e2, 6.9983e1, 2.7039e1, 7.4344e1, 6.3173, 6.5529e1, 5.2381e2)
        mu_r4 = mu_r ** 4.0
        B = [a[k] + b[k] * acentric_factor + c[k] * mu_r4 + d[k] * kappa for k in range(7)]
        y = rho * Vc / 6.0
        G_1 = (1.0 - 0.5 * y) / ((1.0 - y) ** 3.0)
        G_2 = (B[0] * ((1.0 - math.exp(-B[3] * y)) / y) + B[1] * G_1 * math.exp(B[4] * y) + B[2] * G_1) / (B[0] * B[3] + B[1] + B[2])
        q = 3.586e-3 * math.sqrt(Tc / M_prime) / (Vc ** (2.0 / 3.0))
        Tr = T / Tc
        alpha = (Cv / GasConstant) - 1.5
        beta = 0.7862 - 0.7109 * acentric_factor + 1.3168 * (acentric_factor ** 2.0)
        Z = 2.0 + 10.5 * Tr * Tr
        psi = 1.0 + alpha * ((0.215 + 0.28288 * alpha - 1.061 * beta + 0.26665 * Z)
                             / (0.6366 + beta * Z + 1.061 * alpha * beta))
        lambda_val = (31.2 * viscosity * psi / M_prime) * (1.0 / G_2 + B[5] * y) + q * B[6] * y * y * math.sqrt(Tr) * G_2
        return float(lambda_val)

    @property
    def thermal_conductivity(self) -> CallableFloat:
        """High-pressure mixture thermal conductivity using Chung dense fluid correlation."""
        return CallableFloat(self.thermalConductivity())

    def thermalConductivity(self) -> float:
        self.computeMixtureParameters()
        tKelvin = self._thermo.temperature
        T_star = tKelvin / self.m_epsilon_over_k_mix
        kg_per_m3_to_mol_per_cm3 = (1.0 / self.m_MW_mix) * 1e-3
        density = self._thermo.density * kg_per_m3_to_mol_per_cm3
        Cv_mix = float(self._thermo.cv_mole) if not callable(self._thermo.cv_mole) else float(self._thermo.cv_mole())
        return self.highPressureThermalConductivity(
            tKelvin, T_star, self.m_MW_mix, density, Cv_mix, self.m_Vc_mix,
            self.m_Tc_mix, self.m_sigma_mix, self.m_acentric_factor_mix,
            self.m_mu_r_mix, self.m_kappa_mix)
'''

def patch_transport_py():
    repo_root = Path(__file__).resolve().parent.parent
    transport_path = repo_root / "cantera_python" / "transport.py"
    content = transport_path.read_text(encoding="utf-8")

    start_marker = "class HighPressureGasTransportBase(MixTransport):"
    end_marker = "class MMCollisionInt:"
    assert start_marker in content, "start_marker not found"
    assert end_marker in content, "end_marker not found"

    idx_start = content.index(start_marker)
    idx_end = content.index(end_marker)

    new_content = content[:idx_start] + HP_CLASSES_CODE + "\n\n" + content[idx_end:]
    transport_path.write_text(new_content, encoding="utf-8")
    print("cantera_python/transport.py successfully patched.")


if __name__ == "__main__":
    patch_transport_py()
