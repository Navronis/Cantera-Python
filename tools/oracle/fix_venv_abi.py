"""Try successive numpy 2.x versions until the oracle wheel imports."""
import subprocess

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

venv = '/tmp/cantera-oracle/venv/bin'
for ver in ['2.2.6', '2.1.3', '2.0.2']:
    print('--- trying numpy', ver)
    print(run(f'{venv}/pip install --quiet numpy=={ver}')[-200:])
    out = run(f'{venv}/python -c "import cantera as ct; print(\'OK\', ct.__version__, ct.__git_commit__)"')
    print(out[-400:])
    if out.startswith('OK'):
        break
