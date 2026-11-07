"""Execute Python port implementation across common differential corpus.
Runs in Windows pure-Python environment.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cantera_python as cp
from cantera_python.state import set_state_hp, set_state_uv, set_state_sp, set_state_sv
from cantera_python.cubic_eos import PengRobinson, RedlichKwongMFTP
from cantera_python import tpx

AUDIT = ROOT / "audit"
DATA = ROOT.parent / "cantera-original" / "data"
TEST_DATA = ROOT.parent / "cantera-original" / "test" / "data"

def run_wave1():
    print("Python Port: Running Wave 1...")
    cases = [json.loads(line) for line in (AUDIT / "corpus_wave1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    # Load phases
    gri = cp.Solution(DATA / "gri30.yaml")
    air = cp.Solution(DATA / "airNASA9.yaml")
    pr = PengRobinson.from_mechanism(TEST_DATA / "thermo-models.yaml", "CO2-PR")
    rk = RedlichKwongMFTP.from_mechanism(TEST_DATA / "co2_RK_example.yaml")

    out_records = []
    for c in cases:
        cid = c["case_id"]
        model = c["model"]
        solver = c["solver"]

        try:
            if model in ("gri30", "airNASA9"):
                sol = gri if model == "gri30" else air
                gas = sol._thermo._state
                tgt_st = gas.at(temperature=c["t_target"], pressure=c["p_target"])
                if solver == "HP":
                    tgt_v1, tgt_v2 = tgt_st.enthalpy_mass, c["p_target"]
                    fn = set_state_hp
                elif solver == "UV":
                    tgt_v1, tgt_v2 = tgt_st.int_energy_mass, 1.0 / tgt_st.density
                    fn = set_state_uv
                elif solver == "SP":
                    tgt_v1, tgt_v2 = tgt_st.entropy_mass, c["p_target"]
                    fn = set_state_sp
                elif solver == "SV":
                    tgt_v1, tgt_v2 = tgt_st.entropy_mass, 1.0 / tgt_st.density
                    fn = set_state_sv

                start_st = gas.at(temperature=c["t_start"], pressure=c["p_start"])
                solved = fn(start_st, tgt_v1, tgt_v2, rtol=1e-9)

                out_records.append({
                    "case_id": cid,
                    "wave": 1,
                    "status": "SUCCESS",
                    "temperature": float(solved.temperature),
                    "pressure": float(solved.pressure),
                    "density": float(solved.density),
                    "enthalpy": float(solved.enthalpy_mass),
                    "int_energy": float(solved.int_energy_mass),
                    "entropy": float(solved.entropy_mass),
                    "cp": float(solved.cp_mass),
                    "cv": float(solved.cv_mass)
                })
            else:  # CO2-PR or CO2-RK
                obj = pr if model == "CO2-PR" else rk
                obj.setState_TP(c["t_target"], c["p_target"])
                if solver == "HP":
                    tgt_v1, tgt_v2 = obj.enthalpy_mass, c["p_target"]
                    fn_name = "setState_HP"
                elif solver == "UV":
                    tgt_v1, tgt_v2 = obj.int_energy_mass, 1.0 / obj.density
                    fn_name = "setState_UV"
                elif solver == "SP":
                    tgt_v1, tgt_v2 = obj.entropy_mass, c["p_target"]
                    fn_name = "setState_SP"
                elif solver == "SV":
                    tgt_v1, tgt_v2 = obj.entropy_mass, 1.0 / obj.density
                    fn_name = "setState_SV"

                obj.setState_TP(c["t_start"], c["p_start"])
                getattr(obj, fn_name)(tgt_v1, tgt_v2, tol=1e-9)

                out_records.append({
                    "case_id": cid,
                    "wave": 1,
                    "status": "SUCCESS",
                    "temperature": float(obj.temperature),
                    "pressure": float(obj.pressure),
                    "density": float(obj.density),
                    "enthalpy": float(obj.enthalpy_mass),
                    "int_energy": float(obj.int_energy_mass),
                    "entropy": float(obj.entropy_mass),
                    "cp": float(obj.cp_mass),
                    "cv": float(obj.cv_mass)
                })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 1,
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    out_path = AUDIT / "raw_python_wave1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Python Wave 1 complete: {len(out_records)} records -> {out_path.name}")

def run_wave2():
    print("Python Port: Running Wave 2...")
    cases = [json.loads(line) for line in (AUDIT / "corpus_wave2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    pr = PengRobinson.from_mechanism(TEST_DATA / "thermo-models.yaml", "CO2-PR")
    rk = RedlichKwongMFTP.from_mechanism(TEST_DATA / "co2_RK_example.yaml")
    phases = {"CO2-PR": pr, "CO2-RK": rk}

    out_records = []
    for c in cases:
        cid = c["case_id"]
        model = c["model"]
        phase = phases[model]
        t = c["temperature"]
        p = c["pressure"]

        try:
            phase.setState_TP(t, p)
            out_records.append({
                "case_id": cid,
                "wave": 2,
                "status": "SUCCESS",
                "temperature": float(phase.temperature),
                "pressure": float(phase.pressure),
                "density": float(phase.density),
                "molar_volume": float(phase.mean_molecular_weight / phase.density),
                "enthalpy": float(phase.enthalpy_mass),
                "int_energy": float(phase.int_energy_mass),
                "entropy": float(phase.entropy_mass),
                "cp": float(phase.cp_mass),
                "cv": float(phase.cv_mass)
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 2,
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    out_path = AUDIT / "raw_python_wave2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Python Wave 2 complete: {len(out_records)} records -> {out_path.name}")

OFFSETS = {
    "Water": (-1.5970805414650720e+07, 3.5199818593423611e+03),
    "CarbonDioxide": (-9.3693028350615799e+06, 2.6341715836312655e+03),
    "Nitrogen": (-4.5974472188034293e+05, 2.4282693624163958e+03),
    "Oxygen": (-4.6626461421166023e+05, 2.0783004885140672e+03),
    "Hydrogen": (-4.4816311513522100e+06, 5.1880892839525332e+03),
    "Methane": (-5.6300845139379026e+06, 4.2269328154547165e+03),
    "Heptane": (-2.2366524267767663e+06, 3.2908203067620075e+03),
    "HFC134a": (-2.0011789732813556e+05, -2.6809005063933728e+02)
}

def shift_pair_target(pair, v1, v2, hoff, soff):
    if pair == "HP": return v1 + hoff, v2
    if pair == "SP": return v1 + soff, v2
    if pair == "UV": return v1 + hoff, v2
    if pair == "ST": return v1 + soff, v2
    if pair == "SV": return v1 + soff, v2
    if pair == "UP": return v1 + hoff, v2
    if pair == "VH": return v1, v2 + hoff
    if pair == "TH": return v1, v2 + hoff
    if pair == "SH": return v1 + soff, v2 + hoff
    return v1, v2

def normalized_fluid_name(name):
    """Normalize campaign IDs such as ``carbon_dioxide`` to class names."""
    return "".join(character.casefold() for character in name if character.isalnum())

PAIR_TARGET_FIELDS = {
    "TV": ("T", "v"), "HP": ("h", "P"), "SP": ("s", "P"),
    "PV": ("P", "v"), "TP": ("T", "P"), "UV": ("u", "v"),
    "ST": ("s", "T"), "SV": ("s", "v"), "UP": ("u", "P"),
    "VH": ("v", "h"), "TH": ("T", "h"), "SH": ("s", "h"),
    "PX": ("P", "quality"), "TX": ("T", "quality"),
}

def run_wave3():
    print("Python Port: Running Wave 3...")
    fluid_classes = {
        "Water": tpx.Water,
        "CarbonDioxide": tpx.CarbonDioxide,
        "Nitrogen": tpx.Nitrogen,
        "Oxygen": tpx.Oxygen,
        "Hydrogen": tpx.Hydrogen,
        "Methane": tpx.Methane,
        "Heptane": tpx.Heptane,
        "HFC134a": tpx.HFC134a
    }

    out_records = []

    # 1. Forward corpus (8,000 cases)
    fwd_cases = [json.loads(line) for line in (AUDIT / "corpus_wave3_forward.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in fwd_cases:
        cid = c["case_id"]
        cls = fluid_classes[c["fluid"]]
        sub = cls()
        hoff, soff = OFFSETS[c["fluid"]]
        sub.m_energy_offset += hoff
        sub.m_entropy_offset += soff
        try:
            sub.Set(tpx.PropertyPair.TP, c["temperature"], c["pressure"])
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FORWARD",
                "status": "SUCCESS",
                "temperature": float(sub.Temp()),
                "pressure": float(sub.P()),
                "density": float(1.0 / sub.v()),
                "enthalpy": float(sub.h()),
                "int_energy": float(sub.u()),
                "entropy": float(sub.s()),
                "gibbs": float(sub.g()),
                "cp": float(sub.cp()),
                "cv": float(sub.cv())
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FORWARD",
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    # 2. Saturation corpus (200 cases)
    sat_cases = [json.loads(line) for line in (AUDIT / "corpus_wave3_saturation.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in sat_cases:
        cid = c["case_id"]
        cls = fluid_classes[c["fluid"]]
        sub = cls()
        hoff, soff = OFFSETS[c["fluid"]]
        sub.m_energy_offset += hoff
        sub.m_entropy_offset += soff
        try:
            # Saturated liquid
            sub.Set(tpx.PropertyPair.TX, c["temperature"], 0.0)
            p_sat = float(sub.P())
            rho_l = float(1.0 / sub.v())
            h_l = float(sub.h())
            s_l = float(sub.s())

            # Saturated vapor
            sub.Set(tpx.PropertyPair.TX, c["temperature"], 1.0)
            rho_v = float(1.0 / sub.v())
            h_v = float(sub.h())
            s_v = float(sub.s())

            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "SATURATION",
                "status": "SUCCESS",
                "temperature": float(c["temperature"]),
                "pressure": p_sat,
                "rho_liq": rho_l,
                "rho_vap": rho_v,
                "h_liq": h_l,
                "h_vap": h_v,
                "s_liq": s_l,
                "s_vap": s_v
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "SATURATION",
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    # 3. Quality corpus (208 cases)
    qual_cases = [json.loads(line) for line in (AUDIT / "corpus_wave3_quality.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in qual_cases:
        cid = c["case_id"]
        cls = fluid_classes[c["fluid"]]
        sub = cls()
        hoff, soff = OFFSETS[c["fluid"]]
        sub.m_energy_offset += hoff
        sub.m_entropy_offset += soff
        try:
            sub.Set(tpx.PropertyPair.PX, c["pressure"], c["quality"])
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "QUALITY",
                "status": "SUCCESS",
                "temperature": float(sub.Temp()),
                "pressure": float(sub.P()),
                "density": float(1.0 / sub.v()),
                "enthalpy": float(sub.h()),
                "entropy": float(sub.s()),
                "quality": float(c["quality"])
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "QUALITY",
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    # 4. Failure corpus (80 cases)
    fail_cases = [json.loads(line) for line in (AUDIT / "corpus_wave3_failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in fail_cases:
        cid = c["case_id"]
        cls = fluid_classes[c["fluid"]]
        sub = cls()
        hoff, soff = OFFSETS[c["fluid"]]
        sub.m_energy_offset += hoff
        sub.m_entropy_offset += soff
        t = c["temperature"]
        p = c["pressure"]
        q = c["quality"]
        try:
            if "Q" in c.get("fail_category", ""):
                sub.Set(tpx.PropertyPair.PX, p, q)
            else:
                sub.Set(tpx.PropertyPair.TP, t, p)
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FAILURE_TEST",
                "status": "SUCCESS",
                "temperature": float(sub.Temp()),
                "pressure": float(sub.P())
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FAILURE_TEST",
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    # 5. Property-Pair Replay Sample (matching native sampler)
    prop_val_path = AUDIT / "solver_wave3_property_pair_validation.jsonl"
    if prop_val_path.exists():
        print("Sampling 1,120 property-pair states for Python runner...")
        pair_records = []
        with open(prop_val_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                pair_records.append(json.loads(line))

        sample_pairs = pair_records[::20]
        normalized_fluid_classes = {
            normalized_fluid_name(name): name for name in fluid_classes
        }
        for idx, sp in enumerate(sample_pairs):
            fluid_id = normalized_fluid_name(sp["fluid"]).upper()
            cid = f"W3_PAIR_{fluid_id}_{sp['pair']}_{idx+1:04d}"
            fluid_name = normalized_fluid_classes.get(
                normalized_fluid_name(sp["fluid"])
            )
            if fluid_name is None:
                raise ValueError(f"Unknown TPX fluid in property-pair corpus: {sp['fluid']}")
            cls = fluid_classes[fluid_name]
            sub = cls()
            hoff, soff = OFFSETS[fluid_name]
            sub.m_energy_offset += hoff
            sub.m_entropy_offset += soff
            p_code = sp["pair"]
            if "target_arguments" in sp:
                target_arguments = sp["target_arguments"]
            elif isinstance(sp.get("target"), list):
                target_arguments = sp["target"]
            else:
                target_arguments = [sp["target"][key]
                                    for key in PAIR_TARGET_FIELDS[p_code]]
            v1, v2 = shift_pair_target(
                p_code, target_arguments[0], target_arguments[1], hoff, soff
            )
            try:
                if "start" in sp and "T" in sp["start"] and "v" in sp["start"]:
                    sub.Set(tpx.PropertyPair.TV, sp["start"]["T"], sp["start"]["v"])
                elif "start" in sp and "T" in sp["start"] and "rho" in sp["start"]:
                    sub.Set(tpx.PropertyPair.TV, sp["start"]["T"], 1.0 / sp["start"]["rho"])
                pair_enum = getattr(tpx.PropertyPair, p_code)
                sub.Set(pair_enum, v1, v2)
                out_records.append({
                    "case_id": cid,
                    "wave": 3,
                    "sub_type": "PROPERTY_PAIR",
                    "pair": p_code,
                    "fluid": sp["fluid"],
                    "status": "SUCCESS",
                    "temperature": float(sub.Temp()),
                    "pressure": float(sub.P()),
                    "density": float(1.0 / sub.v()),
                    "enthalpy": float(sub.h()),
                    "entropy": float(sub.s())
                })
            except Exception as e:
                out_records.append({
                    "case_id": cid,
                    "wave": 3,
                    "sub_type": "PROPERTY_PAIR",
                    "pair": p_code,
                    "fluid": sp["fluid"],
                    "status": "FAILURE",
                    "error_type": type(e).__name__,
                    "error_msg": str(e)
                })

    out_path = AUDIT / "raw_python_wave3.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Python Wave 3 complete: {len(out_records)} records -> {out_path.name}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "1" in arg:
            run_wave1()
        elif "2" in arg:
            run_wave2()
        elif "3" in arg:
            run_wave3()
        else:
            run_wave1()
            run_wave2()
            run_wave3()
    else:
        run_wave1()
        run_wave2()
        run_wave3()
