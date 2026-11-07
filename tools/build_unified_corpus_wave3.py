import json
from pathlib import Path

audit = Path(__file__).resolve().parents[1] / "audit"
nat = [json.loads(l) for l in (audit / "raw_native_wave3.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

fwd = {c["case_id"]: c for c in [json.loads(l) for l in (audit / "corpus_wave3_forward.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
sat = {c["case_id"]: c for c in [json.loads(l) for l in (audit / "corpus_wave3_saturation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
qual = {c["case_id"]: c for c in [json.loads(l) for l in (audit / "corpus_wave3_quality.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}
fail = {c["case_id"]: c for c in [json.loads(l) for l in (audit / "corpus_wave3_failures.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]}

prop_val_path = audit / "solver_wave3_property_pair_validation.jsonl"
pair_records = []
if prop_val_path.exists():
    with open(prop_val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): pair_records.append(json.loads(line))
sample_pairs = pair_records[::20]
pair_dict = {}
for idx, sp in enumerate(sample_pairs):
    fluid_name = sp["fluid"].upper()
    pair_name = sp["pair"]
    cid = f"W3_PAIR_{fluid_name}_{pair_name}_{idx+1:04d}"
    pair_dict[cid] = {
        "case_id": cid,
        "wave": 3,
        "type": "TPX_PROPERTY_PAIR",
        "fluid": sp["fluid"],
        "pair": sp["pair"],
        "target": sp["target"],
        "start": sp.get("start", {}),
        "expected_success": True
    }

corpus_w3 = []
for r in nat:
    cid = r["case_id"]
    if cid in fwd: corpus_w3.append(fwd[cid])
    elif cid in sat: corpus_w3.append(sat[cid])
    elif cid in qual: corpus_w3.append(qual[cid])
    elif cid in fail: corpus_w3.append(fail[cid])
    elif cid in pair_dict: corpus_w3.append(pair_dict[cid])
    else:
        corpus_w3.append({"case_id": cid, "wave": 3, "type": r.get("sub_type", "UNKNOWN")})

with open(audit / "corpus_wave3.jsonl", "w", encoding="utf-8") as f:
    for c in corpus_w3:
        f.write(json.dumps(c) + "\n")

print(f"Wrote unified corpus_wave3.jsonl with {len(corpus_w3)} cases")
assert len(corpus_w3) == len(nat) == 9608
