"""Differential test for Python VCS phase-stability successive substitution."""
from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cantera_python as ct
from cantera_python.vcs import MultiPhase, VCS_SOLVE

parser = argparse.ArgumentParser()
parser.add_argument('--native', type=Path,
                    default=Path('audit/current_tree_parity/native_vcs_phase_stability_gri.txt'))
path = parser.parse_args().native
setup = None; species={}; dg={}; rxn_species={}; nu={}; local={}; result=None; output={}
for line in path.read_text(encoding='utf-8').splitlines():
    f=line.split()
    if not f: continue
    if f[0]=='setup': setup=tuple(map(int,f[1:]))
    elif f[0]=='sp': species[int(f[1])] = (f[2],int(f[3]),float(f[4]),float(f[5]))
    elif f[0]=='dg': dg[int(f[1])] = float(f[2])
    elif f[0]=='rxn_species': rxn_species[int(f[1])] = int(f[2])
    elif f[0]=='nu': nu[(int(f[1]),int(f[2]))] = float(f[3])
    elif f[0]=='local': local[int(f[1])] = (int(f[2]),float(f[3]),int(f[4]))
    elif f[0]=='result': result=float(f[1])
    elif f[0]=='out': output[int(f[1])] = tuple(map(float,f[2:]))
assert setup and result is not None and len(output)==setup[3]
ncomp,nsp,nrxn,nlocal=setup
native_names=[species[k][0] for k in range(nsp)]
assert native_names == ['N2','C(gr)','CH4','O2','AR'] + native_names[5:]
# Recreate the same native phase state and obtain an independently built Python
# VCS phase object; then impose the exact pinned solver's local/global map.
gas=ct.Solution('gri30.yaml'); gas.TPX=1200.0,ct.OneAtm,'CH4:1, O2:2, N2:7.52'
graphite=ct.Solution('graphite.yaml'); graphite.TP=gas.TP
mix=MultiPhase(); mix.add_phase(gas,1.0); mix.add_phase(graphite,.2)
solver=VCS_SOLVE(); prepared=solver.prepare_problem(mix)
phase=prepared['phase_objects'][0]
assert phase.nSpecies()==nlocal
phase._species_global_index=[local[k][0] for k in range(nlocal)]
phase.setCreationMoleNumbers([local[k][1] for k in range(nlocal)],
                             [(-1 if local[k][2] < 0 else local[k][2]) for k in range(nlocal)])
phase.setTotalMoles(0.0)
fe=np.asarray([species[k][2] for k in range(nsp)])
moles=np.asarray([species[k][3] for k in range(nsp)])
delta=np.asarray([dg[k] for k in range(nrxn)])
stoich=np.zeros((ncomp,nrxn))
for (i,j),v in nu.items(): stoich[i,j]=v
activity=np.ones(nsp)
actual=solver.vcs_phase_stability_test(
    0,delta,fe,moles,activity_coefficients=activity)
print('native result',format(result,'.17g'))
print('python result',format(actual['func_phase_stability'],'.17g'))
print('converged',actual['converged'],'iterations',actual['iterations'])
print('max X error',max(abs(phase.moleFraction(k)-output[k][0]) for k in range(nlocal)))
print('max creation error',max(abs(phase.creationMoleNumbers(k)-output[k][1]) for k in range(nlocal)))
print('max gamma error',max(abs(actual['activity_coefficients'][local[k][0]]-output[k][2]) for k in range(nlocal)))
assert abs(actual['func_phase_stability']-result) <= 1e-12
for k in range(nlocal):
    assert abs(phase.moleFraction(k)-output[k][0]) <= 1e-12, (k,phase.moleFraction(k),output[k][0])
    assert abs(phase.creationMoleNumbers(k)-output[k][1]) <= 1e-12, (k,phase.creationMoleNumbers(k),output[k][1])
    assert abs(actual['activity_coefficients'][local[k][0]]-output[k][2]) <= 1e-14
print('Pinned native VCS phase-stability differential: PASS')
