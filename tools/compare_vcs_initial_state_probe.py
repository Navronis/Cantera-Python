"""Compare pure-Python OLD VCS state construction with pinned Cantera."""
from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cantera_python as ct
from cantera_python.vcs import MultiPhase,VCS_SOLVE

parser=argparse.ArgumentParser()
parser.add_argument('--native',type=Path,default=Path('audit/current_tree_parity/native_vcs_initial_state_gri.txt'))
args=parser.parse_args()
records=args.native.read_text(encoding='utf-8').splitlines()
dims=None; species={}; phases={}; reactions={}
for line in records:
 f=line.split()
 if not f: continue
 if f[0]=='dims': dims=tuple(map(int,f[1:]))
 elif f[0]=='sp': species[int(f[1])]={'name':f[2],'standard':float(f[3]),'ln_mnaught':float(f[4]),'moles':float(f[5]),'status':int(f[6]),'charge':float(f[7]),'phase':int(f[8]),'unknown':int(f[9]),'activity':float(f[10]),'free':float(f[11])}
 elif f[0]=='phase': phases[int(f[1])]={'moles':float(f[2]),'potential':float(f[3])}
 elif f[0]=='dg': reactions[int(f[1])]=float(f[2])
if dims is None: raise ValueError('missing native VCS dimensions')
nsp,ncomp,nrxn,nphase=dims
if len(species)!=nsp or len(phases)!=nphase or len(reactions)!=nrxn: raise ValueError('incomplete native VCS fixture')
gas=ct.Solution('gri30.yaml'); gas.TPX=1200.0,ct.OneAtm,'CH4:1,O2:2,N2:7.52'
graphite=ct.Solution('graphite.yaml'); graphite.TP=gas.TP
mixture=MultiPhase();mixture.add_phase(gas,1.0);mixture.add_phase(graphite,0.2)
solver=VCS_SOLVE(); prepared=solver.prepare_problem(mixture); py=solver.initialize_thermodynamic_state()
if len(prepared['basis_ordered_species_names'])!=nsp: raise AssertionError('species count differs')
for k,name in enumerate(prepared['basis_ordered_species_names']):
 if name!=species[k]['name']: raise AssertionError(f'species ordering differs at {k}: {name} != {species[k]["name"]}')
fields=[('standard_gibbs_rt','standard'),('ln_mnaught_species','ln_mnaught'),('mole_numbers_old','moles'),('species_status','status'),('basis_ordered_species_charges','charge'),('basis_ordered_species_phase_ids','phase'),('basis_ordered_species_unknown_types','unknown'),('activity_coefficients_old','activity'),('species_free_energies_old','free')]
errors={}
for pykey,nativekey in fields:
 actual=np.asarray(py[pykey] if pykey in py else prepared[pykey],dtype=float)
 expected=np.asarray([species[k][nativekey] for k in range(nsp)],dtype=float)
 errors[pykey]=float(np.max(np.abs(actual-expected),initial=0.0))
 if errors[pykey]>2e-12: raise AssertionError(f'{pykey} mismatch: max abs {errors[pykey]:.17g}')
expected_dg=np.asarray([reactions[k] for k in range(nrxn)])
errors['delta_g_rxn_old']=float(np.max(np.abs(py['delta_g_rxn_old']-expected_dg),initial=0.0))
if errors['delta_g_rxn_old']>2e-12: raise AssertionError('formation-reaction Gibbs changes differ')
expected_phase=np.asarray([phases[i]['moles'] for i in range(nphase)])
errors['phase_moles_old']=float(np.max(np.abs(py['phase_moles_old']-expected_phase),initial=0.0))
if errors['phase_moles_old']>2e-14: raise AssertionError('phase mole totals differ')
expected_phi=np.asarray([phases[i]['potential'] for i in range(nphase)])
errors['phase_potentials_old']=float(np.max(np.abs(py['phase_potentials_old']-expected_phi),initial=0.0))
if errors['phase_potentials_old']>2e-14: raise AssertionError('phase potentials differ')
print(f'Pinned native OLD-state construction: PASS ({nsp} species, {nphase} phases, {nrxn} reactions)')
for key,value in errors.items(): print(f'{key}: max abs {value:.17g}')
