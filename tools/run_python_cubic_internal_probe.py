import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cantera_python.cubic_eos import PengRobinson, RedlichKwongMFTP

AUDIT = ROOT / "audit"
CANTERA_ORIG = ROOT.parent / "cantera-original"
TEST_DATA = CANTERA_ORIG / "test" / "data"

def main():
    corpus_path = AUDIT / "corpus_wave2.jsonl"
    out_path = AUDIT / "raw_python_wave2_internal.jsonl"

    print("Loading Python models...")
    pr = PengRobinson.from_mechanism(TEST_DATA / "thermo-models.yaml", "CO2-PR")
    rk = RedlichKwongMFTP.from_mechanism(TEST_DATA / "co2_RK_example.yaml")
    phases = {"CO2-PR": pr, "CO2-RK": rk}

    cases = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Running Python Wave 2 Internal Cubic Probe across {len(cases)} cases...")

    out_records = []
    for c in cases:
        cid = c["case_id"]
        model = c["model"]
        phase = phases[model]
        t = c["temperature"]
        p = c["pressure"]

        try:
            phase.setState_TP(t, p)
            mv = float(phase.mean_molecular_weight / phase.density)
            rho = float(phase.density)

            vroot = [0.0, 0.0, 0.0]
            if model == "CO2-PR":
                nsol = phase.solveCubic(t, p, phase._a, phase._b, phase._aAlpha_mix, vroot)
            elif model == "CO2-RK":
                nsol = phase.solveCubic(t, p, phase._aMix, phase._bMix, vroot)
            else:
                raise ValueError(f"Unknown model: {model}")

            out_records.append({
                "case_id": cid,
                "wave": 2,
                "model": model,
                "status": "SUCCESS",
                "nsol": int(nsol),
                "vroots": [float(v) for v in vroot],
                "molar_volume": mv,
                "density": rho
            })
        except Exception as e:
            out_records.append({
                "case_id": cid,
                "wave": 2,
                "model": model,
                "status": "FAILURE",
                "error_type": type(e).__name__,
                "error_msg": str(e)
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print(f"Python Wave 2 Internal Cubic Probe completed: {len(out_records)} records -> {out_path.name}")

if __name__ == "__main__":
    main()
