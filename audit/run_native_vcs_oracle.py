
import json
import cantera as ct

with open('/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/corpus_vcs.jsonl') as f:
    cases = [json.loads(line) for line in f]

results = {}
for c in cases:
    try:
        phases = []
        for p_info in c['phases']:
            sol = ct.Solution(p_info['source'])
            sol.TP = c['T'], c['P']
            if p_info['name'] == 'gas':
                sol.X = c['X']
            phases.append((sol, p_info['moles']))
        mix = ct.Mixture(phases)
        mix.T = c['T']
        mix.P = c['P']
        mix.equilibrate(c['mode'], solver='vcs')

        pmoles = [float(mix.phase_moles(i)) for i in range(mix.n_phases)]
        gas = phases[0][0]
        key_species = ['H2', 'H2O', 'O2', 'CO', 'CO2', 'CH4', 'OH', 'H', 'O']
        mf = {}
        for sp in key_species:
            if sp in gas.species_names:
                mf[sp] = float(gas.X[gas.species_index(sp)])
        results[c['case_id']] = {
            'status': 'CONVERGED',
            'phase_moles': pmoles,
            'mole_fractions': mf,
            'T': float(mix.T),
            'P': float(mix.P)
        }
    except Exception as e:
        results[c['case_id']] = {'status': 'FAILED', 'error': str(e)}

with open('/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-python/audit/raw_native_vcs.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Native oracle execution finished successfully')
