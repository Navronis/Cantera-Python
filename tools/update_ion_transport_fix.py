import sys

with open('outputs/cantera-python/cantera_python/transport.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = '''    def fitDiffCoeffs(self, integrals=None) -> None:
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

    @property
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
        return self.electricalConductivity()'''

replacement = '''    def fitDiffCoeffs(self, integrals=None) -> None:
        """Fit polynomial coefficients to binary diffusion for ion-neutral pairs.
        
        Matches native Cantera IonGasTransport::fitDiffCoeffs:
        Samples 50 logarithmically spaced temperatures between minTemp and maxTemp,
        computes unit-pressure D_kj / T^1.5 using the Stockmayer-(n,6,4) model
        (or empirical O2/O2- fit), and fits degree-4 polynomials in log(T) with
        weights 1/diff^2.
        """
        np_pts = 50
        degree = 4
        t_min = float(self._thermo.min_temp)
        t_max = float(self._thermo.max_temp)
        dt = (t_max - t_min) / (np_pts - 1)
        t_vals = np.array([t_min + dt * n for n in range(np_pts)], dtype=float)
        tlog = np.log(t_vals)
        mw = np.asarray(self._thermo.molecular_weights, dtype=float)

        self._ion_diff_polys = {}
        for i in self._ions:
            for j in self._neutrals:
                sigma = self._diam_n64[i, j]
                epsilon = self._epsilon_n64[i, j]
                gamma = self._gamma_n64[i, j]
                if sigma <= 0.0 or epsilon <= 0.0:
                    continue
                m_ij = (mw[i] * mw[j]) / ((mw[i] + mw[j]) * Avogadro)
                name_i = self._thermo.species_name(i)
                name_j = self._thermo.species_name(j)

                diff = np.zeros(np_pts, dtype=float)
                for n, t in enumerate(t_vals):
                    if (name_i in ('O2-', 'O2^-') and name_j == 'O2') or (name_j in ('O2-', 'O2^-') and name_i == 'O2'):
                        c_p = self._om11_O2
                        om11 = (((((c_p[5]*t + c_p[4])*t + c_p[3])*t + c_p[2])*t + c_p[1])*t + c_p[0]) / 1e20
                    else:
                        tstar = Boltzmann * t / epsilon
                        om11 = self.omega11_n64(tstar, gamma) * Pi * (sigma**2)
                    diffcoeff = 3.0 / 16.0 * math.sqrt(2.0 * Pi / m_ij) * ((Boltzmann * t)**1.5) / om11
                    diff[n] = diffcoeff / (t**1.5)
                coeffs = _polyfit_householder_pivoted(tlog, diff, degree, weights=1.0 / np.abs(diff))
                self._ion_diff_polys[(i, j)] = coeffs
                self._ion_diff_polys[(j, i)] = coeffs

    def _ion_neutral_diffusion(self, ion, neutral):
        """Evaluate fitted polynomial binary diffusion coefficient [m^2/s] for ion-neutral pair."""
        poly = getattr(self, '_ion_diff_polys', {}).get((ion, neutral))
        if poly is not None:
            T = float(self._thermo.temperature)
            P = float(self._thermo.pressure)
            log_t = math.log(T)
            val = poly[0] + log_t * (poly[1] + log_t * (poly[2] + log_t * (poly[3] + log_t * poly[4])))
            return (T**1.5) * val / P

        # Fallback to direct Stockmayer-(n,6,4) evaluation if not fitted
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
        matrix = np.array(super().binary_diffusion_coefficients(), dtype=float)
        for ion in self._ions:
            for neutral in self._neutrals:
                value = self._ion_neutral_diffusion(ion, neutral)
                matrix[ion][neutral] = matrix[neutral][ion] = value
        return matrix

    @property
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
        x = np.maximum(self._thermo.X, 1.e-20)
        temperature = float(self._thermo.temperature)
        p = float(self._thermo.pressure)
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        if self._electron is not None:
            values[self._electron] = 0.4
        for ion in self._ions:
            inverse = 0.0
            for neutral in self._neutrals:
                bmobi = binary[ion][neutral] * p * ElectronCharge / (Boltzmann * temperature)
                if bmobi > 0.0:
                    inverse += x[neutral] / bmobi
            values[ion] = 1.0 / inverse / p if inverse > 0.0 else 0.0
        if mobi is not None:
            mobi[:] = values
            return mobi
        return values

    def mobilities(self):
        return tuple(self.getMobilities())

    def getMixDiffCoeffs(self, d=None):
        _check_array_size("IonGasTransport::getMixDiffCoeffs", "d", d, self._thermo.n_species) if d is not None else None
        x = np.maximum(self._thermo.X, 1.e-20)
        mw = self._thermo.molecular_weights
        mean_mw = float(self._thermo.mean_molecular_weight)
        p = float(self._thermo.pressure)
        binary = self.binary_diffusion_coefficients()
        values = np.zeros(self._thermo.n_species, dtype=float)
        for k in range(len(x)):
            if k == self._electron:
                values[k] = 0.4 * Boltzmann * float(self._thermo.temperature) / ElectronCharge
                continue
            inverse = sum(x[j] / (p * binary[j][k]) for j in self._neutrals if j != k and binary[j][k] > 0.0)
            if inverse <= 0.0:
                values[k] = binary[k][k]
            else:
                values[k] = (mean_mw - x[k] * mw[k]) / (p * mean_mw * inverse)
        if d is not None:
            d[:] = values
            return d
        return values

    def mix_diff_coeffs(self):
        return tuple(self.getMixDiffCoeffs())

    def electricalConductivity(self) -> CallableFloat:
        x = np.maximum(self._thermo.X, 1.e-20)
        mobi = np.zeros(self._thermo.n_species, dtype=float)
        self.getMobilities(mobi)
        kbt = Boltzmann * float(self._thermo.temperature)
        p = float(self._thermo.pressure)
        nd_factor = p / kbt
        sum_elec = 0.0
        for ion in self._ions:
            sum_elec += x[ion] * nd_factor * abs(self._charges[ion]) * ElectronCharge * mobi[ion]
        if self._electron is not None:
            sum_elec += x[self._electron] * nd_factor * ElectronCharge * mobi[self._electron]
        return CallableFloat(sum_elec)

    @property
    def electrical_conductivity(self) -> CallableFloat:
        return self.electricalConductivity()'''

if target not in content:
    print('ERROR: target not found!')
    sys.exit(1)

new_content = content.replace(target, replacement, 1)
with open('outputs/cantera-python/cantera_python/transport.py', 'w', encoding='utf-8') as f:
    f.write(new_content)
print('SUCCESS: transport.py updated!')
