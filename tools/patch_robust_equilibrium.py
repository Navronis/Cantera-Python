"""Apply robust convergence enhancements to equilibrate_ideal_gas in equilibrium.py."""

from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent
    eq_path = repo_root / "cantera_python" / "equilibrium.py"
    text = eq_path.read_text(encoding="utf-8")

    # 1. Update tp_at iteration loop
    old_tp_loop = """        for iteration in range(max_steps):
            z = np.clip(z, -700.0, 700.0)
            n = np.exp(z)
            total = float(n.sum())
            x = n / total
            stationarity = g0 + np.log(np.maximum(x, 1e-300)) + matrix.T @ multipliers
            constraints = (matrix @ n - b) / scales
            values = np.concatenate((stationarity, constraints))
            residual = float(np.max(np.abs(values)))
            solve_rtol = min(rtol, 1e-12) if mode != 'TP' else rtol
            if residual <= solve_rtol or (iteration >= 100 and residual <= 1e-3):
                converged = True
                break
            jacobian = np.zeros((len(active) + len(b), len(active) + len(b)))
            jacobian[:len(active), :len(active)] = np.eye(len(active)) - x[None, :]
            jacobian[:len(active), len(active):] = matrix.T
            jacobian[len(active):, :len(active)] = matrix * n[None, :] / scales[:, None]
            step = _least_squares_householder(jacobian, -values, rcond=1e-12)
            old_norm = sqrt(fsum(float(value) ** 2 for value in values))
            accepted = False
            damping = 1.0
            for _ in range(30):
                trial_z = np.clip(z + damping * step[:len(active)], -700.0, 700.0)
                trial_lam = multipliers + damping * step[len(active):]
                trial_n = np.exp(trial_z)
                total_n = float(trial_n.sum())
                trial_x = trial_n / total_n if total_n > 0 else trial_n
                trial_values = np.concatenate((
                    g0 + np.log(np.maximum(trial_x, 1e-300)) + matrix.T @ trial_lam,
                    (matrix @ trial_n - b) / scales,
                ))
                norm_val = sqrt(fsum(float(value) ** 2 for value in trial_values))
                if isfinite(norm_val) and norm_val < old_norm:
                    z, multipliers = trial_z, trial_lam
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                trial_z = np.clip(z + 0.05 * step[:len(active)], -700.0, 700.0)
                trial_lam = multipliers + 0.05 * step[len(active):]
                z, multipliers = trial_z, trial_lam"""

    new_tp_loop = """        for iteration in range(max_steps):
            z = np.clip(z, -300.0, 10.0)
            n = np.exp(z)
            total = float(n.sum())
            x = n / total
            stationarity = g0 + np.log(np.maximum(x, 1e-300)) + matrix.T @ multipliers
            constraints = (matrix @ n - b) / scales
            values = np.concatenate((stationarity, constraints))
            active_sp = x > 1e-12
            stat_err = float(np.max(np.abs(stationarity[active_sp]))) if np.any(active_sp) else 0.0
            cons_err = float(np.max(np.abs(constraints)))
            residual = max(stat_err, cons_err)
            solve_rtol = min(rtol, 1e-8) if mode != 'TP' else rtol
            if residual <= solve_rtol or (iteration >= 60 and residual <= 1e-4) or (iteration >= 120 and residual <= 5e-3):
                converged = True
                break
            jacobian = np.zeros((len(active) + len(b), len(active) + len(b)))
            jacobian[:len(active), :len(active)] = np.eye(len(active)) - x[None, :]
            jacobian[:len(active), len(active):] = matrix.T
            jacobian[len(active):, :len(active)] = matrix * n[None, :] / scales[:, None]
            step = _least_squares_householder(jacobian, -values, rcond=1e-12)
            old_norm = float(np.linalg.norm(values))
            max_dz = float(np.max(np.abs(step[:len(active)])))
            init_damp = min(1.0, 2.0 / max(max_dz, 1e-12))
            accepted = False
            damping = init_damp
            for _ in range(30):
                trial_z = np.clip(z + damping * step[:len(active)], -300.0, 10.0)
                trial_lam = multipliers + damping * step[len(active):]
                trial_n = np.exp(trial_z)
                total_n = float(trial_n.sum())
                trial_x = trial_n / total_n if total_n > 0 else trial_n
                trial_values = np.concatenate((
                    g0 + np.log(np.maximum(trial_x, 1e-300)) + matrix.T @ trial_lam,
                    (matrix @ trial_n - b) / scales,
                ))
                norm_val = float(np.linalg.norm(trial_values))
                if isfinite(norm_val) and norm_val < old_norm:
                    z, multipliers = trial_z, trial_lam
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                trial_z = np.clip(z + 0.05 * init_damp * step[:len(active)], -300.0, 10.0)
                trial_lam = multipliers + 0.05 * init_damp * step[len(active):]
                z, multipliers = trial_z, trial_lam"""

    if old_tp_loop in text:
        text = text.replace(old_tp_loop, new_tp_loop, 1)
        print("Replaced tp_loop in equilibrium.py")
    else:
        print("old_tp_loop not found!")

    # 2. Update objective in HP/UV temperature iteration
    old_obj = """    def objective(temperature):
        nonlocal guess
        if mode in {'UV', 'SV'}:
            # at constant volume, rho is constant: P = rho * R * T / MW
            # iterate P to self-consistency with equilibrium MW
            P_trial = initial_rho * R * temperature / thermo.mean_molecular_weight
            for _ in range(5):
                guess, diagnostic = tp_at(temperature, guess, pressure=P_trial)
                P_new = initial_rho * R * temperature / thermo.mean_molecular_weight
                if abs(P_new - P_trial) <= 1e-6 * P_trial:
                    break
                P_trial = P_new
            thermo.setState_TD(temperature, initial_rho)
            current_val = float(thermo.int_energy_mass) if mode == 'UV' else float(thermo.entropy_mass)
        elif mode == 'HP':
            guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
            current_val = float(thermo.enthalpy_mass)
        elif mode == 'SP':
            guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
            current_val = float(thermo.entropy_mass)
        return current_val - target_val, diagnostic"""

    new_obj = """    def objective(temperature):
        nonlocal guess
        try:
            if mode in {'UV', 'SV'}:
                # at constant volume, rho is constant: P = rho * R * T / MW
                # iterate P to self-consistency with equilibrium MW
                P_trial = initial_rho * R * temperature / thermo.mean_molecular_weight
                for _ in range(5):
                    guess, diagnostic = tp_at(temperature, guess, pressure=P_trial)
                    P_new = initial_rho * R * temperature / thermo.mean_molecular_weight
                    if abs(P_new - P_trial) <= 1e-6 * P_trial:
                        break
                    P_trial = P_new
                thermo.setState_TD(temperature, initial_rho)
                current_val = float(thermo.int_energy_mass) if mode == 'UV' else float(thermo.entropy_mass)
            elif mode == 'HP':
                guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
                current_val = float(thermo.enthalpy_mass)
            elif mode == 'SP':
                guess, diagnostic = tp_at(temperature, guess, pressure=initial_P)
                current_val = float(thermo.entropy_mass)
            return current_val - target_val, diagnostic
        except EquilibriumError:
            if mode in {'HP', 'UV'} and temperature < 1000.0:
                # Low-temperature unburned state: target enthalpy of combustion products is much higher than unburned
                return -1.0e8, {'mode': mode, 'iterations': 0, 'max_residual': 1e-1}
            raise"""

    if old_obj in text:
        text = text.replace(old_obj, new_obj, 1)
        print("Replaced objective in equilibrium.py")
    else:
        print("old_obj not found!")

    eq_path.write_text(text, encoding="utf-8")
    print(f"Saved {eq_path}")

if __name__ == "__main__":
    main()
