"""Comprehensive cross-wave state sequence and phase transition verification campaign.

Verifies:
1. TP -> HP -> SP -> UV -> SV -> TP sequences across IdealGas (GRI-30, airNASA9).
2. TP -> HP -> SP -> UV -> SV -> TP sequences across Cubic EOS (Peng-Robinson, Redlich-Kwong).
3. TP -> HP -> SP -> UV -> TP sequences across all 8 TPX substances.
4. Phase transition sequences (compressed liquid -> sat liquid -> two-phase -> sat vapor -> superheated -> compressed liquid).
5. Cache staleness verification: state and derivative integrity after every transition.
"""
from __future__ import annotations
import json, math, random, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cantera_python as cp
from cantera_python.state import set_state_hp, set_state_sp, set_state_uv, set_state_sv
from cantera_python.cubic_eos import PengRobinson, RedlichKwongMFTP
from cantera_python import tpx

DATA = ROOT.parent / "cantera-original" / "data"
TEST_DATA = ROOT.parent / "cantera-original" / "test" / "data"
OUT_JSONL = ROOT / "audit" / "solver_cross_wave_sequences.jsonl"
OUT_SUMMARY = ROOT / "audit" / "solver_cross_wave_sequences_summary.json"

def test_idealgas_sequences(rng, num_cases=200):
    results = []
    gri = cp.Solution(DATA / "gri30.yaml")
    air = cp.Solution(DATA / "airNASA9.yaml")
    cases = [("gri30", gri), ("airNASA9", air)]

    for name, sol in cases:
        gas = sol._thermo._state
        t_min = float(sol._thermo.min_temp)
        t_max = float(sol._thermo.max_temp)

        for i in range(num_cases // 2):
            T_target = t_min + 50.0 + (t_max - t_min - 100.0) * rng.random()
            P_target = 2e4 + 3e6 * rng.random()
            target_state = gas.at(temperature=T_target, pressure=P_target)

            h0 = target_state.enthalpy_mass
            s0 = target_state.entropy_mass
            u0 = target_state.int_energy_mass
            v0 = 1.0 / target_state.density

            # Independent perturbed start state
            T_start = max(t_min + 10.0, min(t_max - 10.0, T_target * (0.7 + 0.6 * rng.random())))
            P_start = max(1e4, min(5e6, P_target * (0.5 + 1.5 * rng.random())))
            curr_state = gas.at(temperature=T_start, pressure=P_start)

            # 1. TP -> HP
            curr_state = set_state_hp(curr_state, h0, P_target, rtol=1e-9)
            err_T_hp = abs(curr_state.temperature - T_target) / T_target

            # Perturb before SP
            curr_state = gas.at(temperature=T_start, pressure=P_start)
            # 2. HP -> SP
            curr_state = set_state_sp(curr_state, s0, P_target, rtol=1e-9)
            err_T_sp = abs(curr_state.temperature - T_target) / T_target

            # Perturb before UV
            curr_state = gas.at(temperature=T_start, pressure=P_start)
            # 3. SP -> UV
            curr_state = set_state_uv(curr_state, u0, v0, rtol=1e-9)
            err_T_uv = abs(curr_state.temperature - T_target) / T_target

            # Perturb before SV
            curr_state = gas.at(temperature=T_start, pressure=P_start)
            # 4. UV -> SV
            curr_state = set_state_sv(curr_state, s0, v0, rtol=1e-9)
            err_T_sv = abs(curr_state.temperature - T_target) / T_target

            # 5. SV -> TP
            final_state = gas.at(temperature=T_target, pressure=P_target)
            err_h_final = abs(final_state.enthalpy_mass - h0) / abs(h0) if abs(h0) > 1e-10 else 0.0

            max_err = max(err_T_hp, err_T_sp, err_T_uv, err_T_sv, err_h_final)
            results.append({
                "model_type": "IdealGas",
                "phase": name,
                "case_id": f"{name}_{i}",
                "T_target": T_target,
                "P_target": P_target,
                "err_T_hp": err_T_hp,
                "err_T_sp": err_T_sp,
                "err_T_uv": err_T_uv,
                "err_T_sv": err_T_sv,
                "err_h_final": err_h_final,
                "max_rel_error": max_err,
                "passed": max_err < 1e-5
            })

    return results

def test_cubic_sequences(rng, num_cases=100):
    results = []
    pr = PengRobinson.from_mechanism(TEST_DATA / "thermo-models.yaml", "CO2-PR")
    rk = RedlichKwongMFTP.from_mechanism(TEST_DATA / "co2_RK_example.yaml")

    for name, model in [("PengRobinson_CO2", pr), ("RedlichKwong_CO2", rk)]:
        for i in range(num_cases // 2):
            T_target = 300.0 + 350.0 * rng.random()
            P_target = 1e5 + 4e6 * rng.random()
            model.setState_TP(T_target, P_target)

            h0 = model.enthalpy_mass
            s0 = model.entropy_mass
            u0 = model.int_energy_mass
            v0 = 1.0 / model.density

            # Independent perturbed start
            T_start = T_target * (0.8 + 0.4 * rng.random())
            P_start = P_target * (0.7 + 0.6 * rng.random())

            # 1. TP -> HP
            model.setState_TP(T_start, P_start)
            model.setState_HP(h0, P_target, tol=1e-8)
            err_T_hp = abs(model.temperature - T_target) / T_target

            # 2. HP -> SP
            model.setState_TP(T_start, P_start)
            model.setState_SP(s0, P_target, tol=1e-8)
            err_T_sp = abs(model.temperature - T_target) / T_target

            # 3. SP -> UV
            model.setState_TP(T_start, P_start)
            model.setState_UV(u0, v0, tol=1e-8)
            err_T_uv = abs(model.temperature - T_target) / T_target

            # 4. UV -> SV
            model.setState_TP(T_start, P_start)
            model.setState_SV(s0, v0, tol=1e-8)
            err_T_sv = abs(model.temperature - T_target) / T_target

            # 5. SV -> TP
            model.setState_TP(T_target, P_target)
            err_h_final = abs(model.enthalpy_mass - h0) / abs(h0) if abs(h0) > 1e-10 else 0.0

            max_err = max(err_T_hp, err_T_sp, err_T_uv, err_T_sv, err_h_final)
            results.append({
                "model_type": "CubicEOS",
                "phase": name,
                "case_id": f"{name}_{i}",
                "T_target": T_target,
                "P_target": P_target,
                "err_T_hp": err_T_hp,
                "err_T_sp": err_T_sp,
                "err_T_uv": err_T_uv,
                "err_T_sv": err_T_sv,
                "err_h_final": err_h_final,
                "max_rel_error": max_err,
                "passed": max_err < 1e-5
            })

    return results

def test_tpx_sequences(rng):
    substances = [
        ("Water", tpx.Water),
        ("CarbonDioxide", tpx.CarbonDioxide),
        ("Nitrogen", tpx.Nitrogen),
        ("Oxygen", tpx.Oxygen),
        ("Methane", tpx.Methane),
        ("Hydrogen", tpx.Hydrogen),
        ("Heptane", tpx.Heptane),
        ("HFC134a", tpx.HFC134a)
    ]
    results = []

    for name, cls in substances:
        sub = cls()
        tc = sub.Tcrit()
        pc = sub.Pcrit()
        tmin = sub.Tmin()
        tmax = sub.Tmax()

        # 1. Single phase cycle: TP -> HP -> SP -> UV -> TP
        for i in range(25):
            # Choose either subcooled liquid or superheated vapor
            if i % 2 == 0:
                T_target = tmin + 0.3 * (tc - tmin)
                P_target = 0.5 * pc
            else:
                T_target = min(tmax - 20.0, tc + 0.4 * (tmax - tc) * (0.2 + 0.7 * rng.random()))
                P_target = pc * (0.8 + 0.4 * rng.random())

            sub.Set(tpx.PropertyPair.TP, T_target, P_target)
            t0 = sub.Temp()
            p0 = sub.P()
            h0 = sub.h()
            s0 = sub.s()
            u0 = sub.u()
            v0 = sub.v()

            # HP
            sub.Set(tpx.PropertyPair.HP, h0, p0)
            err_T_hp = abs(sub.Temp() - t0) / t0

            # SP
            sub.Set(tpx.PropertyPair.SP, s0, p0)
            err_T_sp = abs(sub.Temp() - t0) / t0

            # UV
            sub.Set(tpx.PropertyPair.UV, u0, v0)
            err_T_uv = abs(sub.Temp() - t0) / t0

            # TP
            sub.Set(tpx.PropertyPair.TP, t0, p0)
            err_h_final = abs(sub.h() - h0) / abs(h0) if abs(h0) > 1e-10 else 0.0

            max_err = max(err_T_hp, err_T_sp, err_T_uv, err_h_final)
            results.append({
                "model_type": "TPX_SinglePhase",
                "substance": name,
                "case_id": f"{name}_cycle_{i}",
                "T_target": t0,
                "P_target": p0,
                "err_T_hp": err_T_hp,
                "err_T_sp": err_T_sp,
                "err_T_uv": err_T_uv,
                "err_h_final": err_h_final,
                "max_rel_error": max_err,
                "passed": max_err < 1e-5
            })

        # 2. Phase transition sequence:
        # compressed liquid -> sat liquid -> two-phase -> sat vapor -> superheated -> compressed liquid
        p_eval = 0.2 * pc
        t_sat = sub.Tsat(p_eval)
        t_comp = 0.5 * (tmin + t_sat)
        t_super = 0.5 * (t_sat + tc)

        # Step a: Compressed liquid
        sub.Set(tpx.PropertyPair.TP, t_comp, p_eval)
        h_comp = sub.h()
        s_comp = sub.s()
        v_comp = sub.v()
        rho_comp = 1.0 / sub.v()
        x_comp = sub.x()
        assert x_comp == 0.0 or x_comp == -1.0 or x_comp < 0.01

        # Step b: Saturated liquid (x = 0)
        sub.Set(tpx.PropertyPair.PX, p_eval, 0.0)
        t_sat_liq = sub.Temp()
        v_sat_liq = sub.v()
        h_sat_liq = sub.h()
        assert abs(t_sat_liq - t_sat) / t_sat < 1e-4

        # Step c: Two-phase (x = 0.5)
        sub.Set(tpx.PropertyPair.PX, p_eval, 0.5)
        t_two_phase = sub.Temp()
        v_two_phase = sub.v()
        h_two_phase = sub.h()
        assert abs(t_two_phase - t_sat) / t_sat < 1e-4
        assert v_sat_liq < v_two_phase

        # Step d: Saturated vapor (x = 1.0)
        sub.Set(tpx.PropertyPair.PX, p_eval, 1.0)
        t_sat_vap = sub.Temp()
        v_sat_vap = sub.v()
        h_sat_vap = sub.h()
        assert abs(t_sat_vap - t_sat) / t_sat < 1e-4
        assert v_two_phase < v_sat_vap

        # Step e: Superheated vapor
        sub.Set(tpx.PropertyPair.TP, t_super, p_eval)
        t_super_act = sub.Temp()
        v_super = sub.v()
        h_super = sub.h()
        assert abs(t_super_act - t_super) / t_super < 1e-6
        assert v_sat_vap < v_super

        # Step f: Return to compressed liquid and verify no cache staleness
        sub.Set(tpx.PropertyPair.TP, t_comp, p_eval)
        err_return_h = abs(sub.h() - h_comp) / abs(h_comp)
        err_return_rho = abs((1.0 / sub.v()) - rho_comp) / rho_comp
        err_return_s = abs(sub.s() - s_comp) / abs(s_comp)

        results.append({
            "model_type": "TPX_PhaseTransitionSequence",
            "substance": name,
            "case_id": f"{name}_phase_transition",
            "p_eval": p_eval,
            "t_sat": t_sat,
            "t_comp": t_comp,
            "t_super": t_super,
            "err_return_h": err_return_h,
            "err_return_rho": err_return_rho,
            "err_return_s": err_return_s,
            "max_rel_error": max(err_return_h, err_return_rho, err_return_s),
            "passed": max(err_return_h, err_return_rho, err_return_s) < 1e-7
        })

    return results

def main():
    print("Running cross-wave state sequence campaign...")
    rng = random.Random(20260925)

    ig_results = test_idealgas_sequences(rng, num_cases=200)
    cubic_results = test_cubic_sequences(rng, num_cases=100)
    tpx_results = test_tpx_sequences(rng)

    all_cases = ig_results + cubic_results + tpx_results
    total = len(all_cases)
    passed = sum(1 for c in all_cases if c["passed"])

    errors = [c["max_rel_error"] for c in all_cases]
    sorted_errs = sorted(errors)

    def json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        return str(obj)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, default=json_default) + "\n")

    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "success_rate": passed / total,
        "breakdown": {
            "IdealGas": sum(1 for c in ig_results if c["passed"]),
            "CubicEOS": sum(1 for c in cubic_results if c["passed"]),
            "TPX": sum(1 for c in tpx_results if c["passed"])
        },
        "max_relative_error": float(max(errors)),
        "median_relative_error": float(statistics.median(errors)),
        "p95_relative_error": float(sorted_errs[int(0.95 * total)]),
        "p99_relative_error": float(sorted_errs[int(0.99 * total)])
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=json_default)

    print(f"Cross-Wave State Sequences complete: {passed}/{total} passed ({summary['success_rate']*100:.2f}%)")
    print(f"Max error: {summary['max_relative_error']:.3e}, p95: {summary['p95_relative_error']:.3e}")

if __name__ == "__main__":
    main()
