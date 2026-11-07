"""Generate deterministic 30-case differential validation corpus for Flow1D / StFlow.

Covers:
- Free flow (1–8)
- Axisymmetric strained flow (9–16)
- Unstrained flow (17–20)
- Two-point control enabled (21–24)
- Fixed temperature mode & energy off (25–26)
- Radiation enabled (27–28)
- Soret effect enabled (29)
- Pseudo-transient rdt > 0 (30)
"""
import json
import numpy as np
from pathlib import Path


def generate_corpus():
    cases = []
    mechs = ["h2o2.yaml"]

    # 10 species in h2o2: H2, H, O, O2, OH, H2O, HO2, H2O2, AR, N2 (if present)
    # Standard h2o2.yaml has: H2, H, O, O2, OH, H2O, HO2, H2O2, AR, HE (10 species)
    species_list = ["H2", "H", "O", "O2", "OH", "H2O", "HO2", "H2O2", "AR", "N2"]

    for i in range(1, 31):
        case_id = f"flow1d_case_{i:02d}"

        # Choose configuration
        if i <= 8:
            domain_type = "free-flow"
        elif i <= 16:
            domain_type = "axisymmetric-flow"
        elif i <= 20:
            domain_type = "unstrained-flow"
        elif i <= 24:
            domain_type = "axisymmetric-flow"
        else:
            domain_type = "free-flow"

        # Grid points
        if i % 3 == 0:
            n_pts = 4
            grid = [0.0, 0.01, 0.025, 0.05]
        elif i % 3 == 1:
            n_pts = 6
            grid = [0.0, 0.005, 0.012, 0.025, 0.04, 0.06]
        else:
            n_pts = 5
            grid = [0.0, 0.01, 0.02, 0.035, 0.05]

        pressure = 101325.0 * (1.0 + 0.5 * (i % 4))
        energy_enabled = not (25 <= i <= 26)
        radiation_enabled = (27 <= i <= 28)
        soret_enabled = (i == 29)
        two_point_control = (21 <= i <= 24)

        z_left = 0.005
        T_left = 600.0
        z_right = grid[-2]
        T_right = 1400.0
        fixed_temperature = 800.0 if domain_type == "free-flow" else None
        rdt = 1000.0 if i == 30 else 0.0

        # State profile at each point
        state_profile = []
        for j in range(n_pts):
            frac = float(j) / float(n_pts - 1)
            T_j = 300.0 + frac * 1800.0
            u_j = 0.5 + frac * 2.0
            V_j = (frac - 0.5) * 10.0 if domain_type == "axisymmetric-flow" else 0.0
            L_j = -50.0 if domain_type == "axisymmetric-flow" else 0.0
            Uo_j = 0.2 if two_point_control else 0.0

            # Species mass fractions
            # Unburned on left (H2: 0.05, O2: 0.2, AR: 0.75), burned on right (H2O: 0.25, AR: 0.75)
            y_H2 = max(0.05 * (1.0 - frac), 1e-6)
            y_O2 = max(0.20 * (1.0 - frac), 1e-6)
            y_H2O = max(0.25 * frac, 1e-6)
            y_OH = max(0.005 * np.sin(np.pi * frac), 1e-6)
            y_H = max(0.001 * np.sin(np.pi * frac), 1e-6)
            y_O = max(0.001 * np.sin(np.pi * frac), 1e-6)
            y_HO2 = max(1e-4 * (1.0 - frac), 1e-6)
            y_H2O2 = max(1e-4 * (1.0 - frac), 1e-6)
            y_sum_other = y_H2 + y_O2 + y_H2O + y_OH + y_H + y_O + y_HO2 + y_H2O2
            y_AR = max(1.0 - y_sum_other, 0.1)

            # Re-normalize
            total = y_H2 + y_O2 + y_H2O + y_OH + y_H + y_O + y_HO2 + y_H2O2 + y_AR
            Y_dict = {
                "H2": float(y_H2 / total),
                "O2": float(y_O2 / total),
                "H2O": float(y_H2O / total),
                "OH": float(y_OH / total),
                "H": float(y_H / total),
                "O": float(y_O / total),
                "HO2": float(y_HO2 / total),
                "H2O2": float(y_H2O2 / total),
                "AR": float(y_AR / total),
            }

            state_profile.append({
                "j": j,
                "z": float(grid[j]),
                "u": float(u_j),
                "V": float(V_j),
                "T": float(T_j),
                "Lambda": float(L_j),
                "Uo": float(Uo_j),
                "Y": Y_dict,
            })

        case = {
            "case_id": case_id,
            "mechanism": "h2o2.yaml",
            "domain_type": domain_type,
            "grid": [float(z) for z in grid],
            "pressure": float(pressure),
            "energy_enabled": energy_enabled,
            "radiation_enabled": radiation_enabled,
            "emissivity_left": 0.15 if radiation_enabled else 0.0,
            "emissivity_right": 0.85 if radiation_enabled else 0.0,
            "soret_enabled": soret_enabled,
            "two_point_control": two_point_control,
            "z_left": z_left,
            "T_left": T_left,
            "z_right": z_right,
            "T_right": T_right,
            "fixed_temperature": fixed_temperature,
            "rdt": float(rdt),
            "state_profile": state_profile,
        }
        cases.append(case)

    out_path = Path("audit/corpus_flow1d.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    print(f"Generated {len(cases)} cases in {out_path}")


if __name__ == "__main__":
    generate_corpus()
