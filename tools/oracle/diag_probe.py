"""Diagnose the scons python_cmd probe failure."""
import subprocess

def run(cmd, tail=800):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()[-tail:]

venv = '/tmp/cantera-oracle/venv/bin'
print(run(f'{venv}/python -c "import sys; print(sys.version)"'))
print('---')
# Reproduce the site_scons probe: it asks python for a JSON blob of build info.
probe = (
    'import json,sysconfig,sys;'
    'print(json.dumps({'
    '"include_dirs": sysconfig.get_paths()["include"],'
    '"library_dirs": sysconfig.get_paths()["purelib"]}))'
)
print(run(f'{venv}/python -c "{probe}"'))
print('---')
print(run(f'{venv}/python -m pip list 2>/dev/null | grep -iE "cython|jinja|numpy|packaging"'))
