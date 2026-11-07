import json
from pathlib import Path
import subprocess
import sys
import unittest


class RuntimePurityTests(unittest.TestCase):
    def test_fresh_process_import_does_not_load_native_cantera(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import json
import sys
import cantera_python
loaded = sorted(name for name in sys.modules
                if name in {"cantera", "_cantera"}
                or name.startswith("cantera."))
print(json.dumps({"loaded_native_modules": loaded,
                  "port_complete": cantera_python.PORT_COMPLETE}))
'''
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=root, check=True,
            capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertEqual(report['loaded_native_modules'], [])
        self.assertFalse(report['port_complete'])

    def test_parity_solve_loads_sundials_but_not_native_cantera(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import json
import sys
from pathlib import Path
import cantera_python as ct
try:
    import sksundae
except ImportError:
    print(json.dumps({"skipped": True}))
    raise SystemExit
gas = ct.Solution(Path(ct.DATA_DIR) / "gri30.yaml")
gas.TPX = 1000.0, ct.OneAtm, "N2:1"
net = ct.ReactorNet(ct.IdealGasReactor(gas), solver_backend="parity")
net.advance(1e-6)
loaded = sorted(name for name in sys.modules
                if name in {"cantera", "_cantera"}
                or name.startswith("cantera."))
print(json.dumps({"skipped": False, "loaded_native_modules": loaded,
                  "sundials_version": sksundae.SUNDIALS_VERSION,
                  "backend": net._last_solver_stats["backend"]}))
'''
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=root, check=True,
            capture_output=True, text=True)
        report = json.loads(result.stdout)
        if report.get('skipped'):
            self.skipTest('optional scikit-sundae parity dependency is absent')
        self.assertEqual(report['loaded_native_modules'], [])
        self.assertEqual(report['sundials_version'], '7.5.0')
        self.assertEqual(report['backend'], 'sksundae-cvode')

    def test_native_python_selection_does_not_load_sundials(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import json
import sys
from pathlib import Path
import cantera_python as ct
gas = ct.Solution(Path(ct.DATA_DIR) / "gri30.yaml")
gas.TPX = 1000.0, ct.OneAtm, "N2:1"
net = ct.ReactorNet(ct.IdealGasReactor(gas), solver_backend="native_python")
net.advance(1e-6)
print(json.dumps({"sksundae_loaded": "sksundae" in sys.modules,
                  "scipy_integrate_loaded": any(name == "scipy.integrate"
                  or name.startswith("scipy.integrate.") for name in sys.modules),
                  "native_loaded": any(name in {"cantera", "_cantera"}
                  or name.startswith("cantera.") for name in sys.modules),
                  "backend": net._last_solver_stats["backend"]}))
'''
        result = subprocess.run(
            [sys.executable, '-c', code], cwd=root, check=True,
            capture_output=True, text=True)
        report = json.loads(result.stdout)
        self.assertFalse(report['sksundae_loaded'])
        self.assertFalse(report['scipy_integrate_loaded'])
        self.assertFalse(report['native_loaded'])
        self.assertEqual(report['backend'], 'in-package-python-cv-bdf')

    def test_runtime_sources_do_not_import_native_cantera(self):
        package = Path(__file__).resolve().parents[1] / 'cantera_python'
        forbidden = []
        for path in package.glob('*.py'):
            for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                stripped = line.strip()
                if stripped == 'import cantera' or stripped.startswith('from cantera '):
                    forbidden.append(f'{path.name}:{line_number}')
        self.assertEqual(forbidden, [])


if __name__ == '__main__':
    unittest.main()
