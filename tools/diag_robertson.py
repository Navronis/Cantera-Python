import json

with open('outputs/cantera-python/audit/idas_integrator_native_oracle_results.jsonl') as fn, open('outputs/cantera-python/audit/idas_integrator_python_port_results.jsonl') as fp:
    for ln, lp in zip(fn, fp):
        dn = json.loads(ln)
        dp = json.loads(lp)
        if dn['case_id'] in ('idas_robertson_stiff_short', 'idas_suppress_alg_false', 'idas_suppress_alg_true'):
            print(f"=== {dn['case_id']} ===")
            for i, (cn, cp) in enumerate(zip(dn['checkpoints'], dp['checkpoints'])):
                print(f"cp {i} t={cn['time']}:")
                print("   native:", cn['y'])
                print("   python:", cp['y'])
                diff = [abs(a - b) for a, b in zip(cn['y'], cp['y'])]
                rel = [abs(a - b) / (abs(a) + 1e-12) for a, b in zip(cn['y'], cp['y'])]
                print("   abs diff:", diff)
                print("   rel diff:", rel)
