"""Background reactor integration check: CV and CP GRI-30 ignition at 1200 K."""
from pathlib import Path
import time
import cantera_python as ct
from cantera_python.reactor import IdealGasReactor, IdealGasConstPressureReactor, ReactorNet
from cantera_python.constants import OneAtm

DATA = Path(__file__).parents[2] / 'cantera-original' / 'data' / 'gri30.yaml'
out = []

sol = ct.Solution(DATA)
sol.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
r = IdealGasReactor(sol)
sim = ReactorNet(r)
t0 = time.time()
sim.advance(1e-5)
cv_t = time.time() - t0
out.append('CV advance %.1fs T=%.2f time=%.3e' % (cv_t, sol.T, sim.time))

sol.TPX = 1200.0, OneAtm, 'CH4:1, O2:2, N2:7.52'
r2 = IdealGasConstPressureReactor(sol)
sim2 = ReactorNet(r2)
t0 = time.time()
sim2.advance(1e-5)
cp_t = time.time() - t0
out.append('CP advance %.1fs T=%.2f time=%.3e' % (cp_t, sol.T, sim2.time))

out.append('DONE')
Path(__file__).with_name('_bg_reactor_result.txt').write_text('\n'.join(out))
print('\n'.join(out))
