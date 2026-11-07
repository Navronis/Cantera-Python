"""Execute exact pinned native Cantera oracle across common differential corpus.
Must run inside WSL with /home/nebula/cantera-oracle-env/bin/python3.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

# Paths in WSL
WSL_ROOT = Path("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python")
AUDIT = WSL_ROOT / "audit"
CANTERA_ORIG = Path("/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original")
DATA = CANTERA_ORIG / "data"
TEST_DATA = CANTERA_ORIG / "test" / "data"

import cantera as ct

def run_wave1():
    print("Native Oracle: Running Wave 1...")
    cases = [json.loads(line) for line in (AUDIT / "corpus_wave1.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    
    # Pre-instantiate phases
    gri = ct.Solution(str(DATA / "gri30.yaml"))
    air = ct.Solution(str(DATA / "airNASA9.yaml"))
    pr = ct.Solution(str(TEST_DATA / "thermo-models.yaml"), "CO2-PR")
    rk = ct.Solution(str(TEST_DATA / "co2_RK_example.yaml"))
    phases = {"gri30": gri, "airNASA9": air, "CO2-PR": pr, "CO2-RK": rk}

    out_records = []
    for c in cases:
        cid = c["case_id"]
        model = c["model"]
        solver = c["solver"]
        phase = phases[model]

        try:
            # 1. Target state evaluation
            phase.TP = c["t_target"], c["p_target"]
            if solver == "HP":
                tgt_v1, tgt_v2 = phase.enthalpy_mass, c["p_target"]
            elif solver == "UV":
                tgt_v1, tgt_v2 = phase.int_energy_mass, 1.0 / phase.density
            elif solver == "SP":
                tgt_v1, tgt_v2 = phase.entropy_mass, c["p_target"]
            elif solver == "SV":
                tgt_v1, tgt_v2 = phase.entropy_mass, 1.0 / phase.density

            # 2. Starting state
            phase.TP = c["t_start"], c["p_start"]

            # 3. Solver execution
            if solver == "HP":
                phase.HP = tgt_v1, tgt_v2
            elif solver == "UV":
                phase.UV = tgt_v1, tgt_v2
            elif solver == "SP":
                phase.SP = tgt_v1, tgt_v2
            elif solver == "SV":
                phase.SV = tgt_v1, tgt_v2

            out_records.append({
                "case_id": cid,
                "wave": 1,
                "status": "SUCCESS",
                "temperature": float(phase.T),
                "pressure": float(phase.P),
                "density": float(phase.density),
                "enthalpy": float(phase.enthalpy_mass),
                "int_energy": float(phase.int_energy_mass),
                "entropy": float(phase.entropy_mass),
                "cp": float(phase.cp_mass),
                "cv": float(phase.cv_mass)
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 1,
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    out_path = AUDIT / "raw_native_wave1.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Native Wave 1 complete: {len(out_records)} records -> {out_path.name}")

def run_wave2():
    print("Native Oracle: Running Wave 2...")
    cases = [json.loads(line) for line in (AUDIT / "corpus_wave2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]

    pr = ct.Solution(str(TEST_DATA / "thermo-models.yaml"), "CO2-PR")
    rk = ct.Solution(str(TEST_DATA / "co2_RK_example.yaml"))
    phases = {"CO2-PR": pr, "CO2-RK": rk}

    out_records = []
    for c in cases:
        cid = c["case_id"]
        model = c["model"]
        phase = phases[model]
        t = c["temperature"]
        p = c["pressure"]

        try:
            phase.TP = t, p
            out_records.append({
                "case_id": cid,
                "wave": 2,
                "status": "SUCCESS",
                "temperature": float(phase.T),
                "pressure": float(phase.P),
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

    out_path = AUDIT / "raw_native_wave2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Native Wave 2 complete: {len(out_records)} records -> {out_path.name}")

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
    """Normalize campaign IDs such as ``carbon_dioxide`` to native names."""
    return "".join(character.casefold() for character in name if character.isalnum())

PAIR_TARGET_FIELDS = {
    "TV": ("T", "v"), "HP": ("h", "P"), "SP": ("s", "P"),
    "PV": ("P", "v"), "TP": ("T", "P"), "UV": ("u", "v"),
    "ST": ("s", "T"), "SV": ("s", "v"), "UP": ("u", "P"),
    "VH": ("v", "h"), "TH": ("T", "h"), "SH": ("s", "h"),
    "PX": ("P", "quality"), "TX": ("T", "quality"),
}

def run_wave3():
    print("Native Oracle: Running Wave 3...")
    fluids = {
        "Water": ct.Water(),
        "CarbonDioxide": ct.CarbonDioxide(),
        "Nitrogen": ct.Nitrogen(),
        "Oxygen": ct.Oxygen(),
        "Hydrogen": ct.Hydrogen(),
        "Methane": ct.Methane(),
        "Heptane": ct.Heptane(),
        "HFC134a": ct.Hfc134a()
    }

    out_records = []

    # 1. Forward corpus (8,000 cases)
    fwd_cases = [json.loads(line) for line in (AUDIT / "corpus_wave3_forward.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for c in fwd_cases:
        cid = c["case_id"]
        fluid = fluids[c["fluid"]]
        try:
            try: fluid.TP = 300.0, 101325.0
            except Exception: pass
            fluid.TP = c["temperature"], c["pressure"]
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FORWARD",
                "status": "SUCCESS",
                "temperature": float(fluid.T),
                "pressure": float(fluid.P),
                "density": float(fluid.density),
                "enthalpy": float(fluid.enthalpy_mass),
                "int_energy": float(fluid.int_energy_mass),
                "entropy": float(fluid.entropy_mass),
                "gibbs": float(fluid.gibbs_mass),
                "cp": float(fluid.cp_mass),
                "cv": float(fluid.cv_mass)
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
        fluid = fluids[c["fluid"]]
        try:
            # Saturated liquid
            fluid.TQ = c["temperature"], 0.0
            p_sat = float(fluid.P)
            rho_l = float(fluid.density)
            h_l = float(fluid.enthalpy_mass)
            s_l = float(fluid.entropy_mass)

            # Saturated vapor
            fluid.TQ = c["temperature"], 1.0
            rho_v = float(fluid.density)
            h_v = float(fluid.enthalpy_mass)
            s_v = float(fluid.entropy_mass)

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
        fluid = fluids[c["fluid"]]
        try:
            fluid.PQ = c["pressure"], c["quality"]
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "QUALITY",
                "status": "SUCCESS",
                "temperature": float(fluid.T),
                "pressure": float(fluid.P),
                "density": float(fluid.density),
                "enthalpy": float(fluid.enthalpy_mass),
                "entropy": float(fluid.entropy_mass),
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
        fluid = fluids[c["fluid"]]
        t = c["temperature"]
        p = c["pressure"]
        q = c["quality"]
        try:
            if "Q" in c.get("fail_category", ""):
                fluid.PQ = p, q
            else:
                fluid.TP = t, p
            out_records.append({
                "case_id": cid,
                "wave": 3,
                "sub_type": "FAILURE_TEST",
                "status": "SUCCESS",
                "temperature": float(fluid.T),
                "pressure": float(fluid.P)
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

    # 5. Property-Pair Replay Sample (1,120 representative states: 14 pairs x 8 fluids x 10 states)
    # Sampling from audit/solver_wave3_property_pair_validation.jsonl
    prop_val_path = AUDIT / "solver_wave3_property_pair_validation.jsonl"
    if prop_val_path.exists():
        print("Sampling 1,120 property-pair states for direct native comparison...")
        pair_records = []
        with open(prop_val_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                rec = json.loads(line)
                pair_records.append(rec)
        
        # Stride sampling 1 in 20 -> 22,400 / 20 = 1,120 cases
        sample_pairs = pair_records[::20]
        for idx, sp in enumerate(sample_pairs):
            fluid_id = normalized_fluid_name(sp["fluid"]).upper()
            cid = f"W3_PAIR_{fluid_id}_{sp['pair']}_{idx+1:04d}"
            normalized_fluids = {normalized_fluid_name(name): name for name in fluids}
            fluid_name = normalized_fluids.get(normalized_fluid_name(sp["fluid"]))
            if fluid_name is None:
                raise ValueError(f"Unknown TPX fluid in property-pair corpus: {sp['fluid']}")
            fluid = fluids[fluid_name]
            p_code = sp["pair"]
            hoff, soff = OFFSETS[fluid_name]
            if "target_arguments" in sp:
                target_arguments = sp["target_arguments"]
            elif isinstance(sp.get("target"), list):
                # Legacy corpus rows stored the setter pair as a two-item list.
                target_arguments = sp["target"]
            else:
                target_arguments = [sp["target"][key]
                                    for key in PAIR_TARGET_FIELDS[p_code]]
            v1, v2 = shift_pair_target(p_code, target_arguments[0],
                                       target_arguments[1], hoff, soff)
            try:
                if "start" in sp and "T" in sp["start"] and "rho" in sp["start"]:
                    fluid.TD = sp["start"]["T"], sp["start"]["rho"]
                elif "start" in sp and "T" in sp["start"] and "v" in sp["start"]:
                    fluid.TD = sp["start"]["T"], 1.0 / sp["start"]["v"]
                else:
                    try: fluid.TP = 300.0, 101325.0
                    except Exception: pass
                # Dispatch pair
                if hasattr(fluid, p_code):
                    setattr(fluid, p_code, (v1, v2))
                elif p_code == "HP": fluid.HP = v1, v2
                elif p_code == "SP": fluid.SP = v1, v2
                elif p_code == "TP": fluid.TP = v1, v2
                elif p_code == "UV": fluid.UV = v1, v2
                elif p_code == "SV": fluid.SV = v1, v2
                elif p_code == "PV": fluid.PV = v1, v2
                elif p_code == "TV": fluid.TV = v1, v2
                elif p_code == "ST": fluid.ST = v1, v2
                elif p_code == "PX": fluid.PQ = v1, v2
                elif p_code == "TX": fluid.TQ = v1, v2
                else:
                    # Generic setter if supported
                    continue

                out_records.append({
                    "case_id": cid,
                    "wave": 3,
                    "sub_type": "PROPERTY_PAIR",
                    "pair": p_code,
                    "fluid": sp["fluid"],
                    "status": "SUCCESS",
                    "temperature": float(fluid.T),
                    "pressure": float(fluid.P),
                    "density": float(fluid.density),
                    "enthalpy": float(fluid.enthalpy_mass),
                    "entropy": float(fluid.entropy_mass)
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

    out_path = AUDIT / "raw_native_wave3.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")
    print(f"Native Wave 3 complete: {len(out_records)} records -> {out_path.name}")

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
