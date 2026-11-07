import sys
sys.path.insert(0, '/home/nebula/cantera-pinned-oracle/build/python')
import cantera as ct

s = ct.Interface('/home/nebula/cantera-pinned-oracle/test/data/blowers-masel.yaml', 'Pt_surf')
print('Native rxn count:', len(s.reactions()))
for i, r in enumerate(s.reactions()):
    print(i, r.equation)
