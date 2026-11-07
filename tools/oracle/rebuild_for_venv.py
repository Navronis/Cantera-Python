"""Install oracle build deps in the venv, rebuild bindings, verify import."""
import subprocess

def run(cmd, tail=500):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()[-tail:]

venv = '/tmp/cantera-oracle/venv/bin'
print(run(f'{venv}/pip install --quiet jinja2 numpy "packaging" "cython>=3.0"'))
print('--- rebuild:')
print(run(
    'cd /tmp/cantera-oracle/src && '
    f'scons build prefix=/tmp/cantera-oracle/install python_cmd={venv}/python -j4',
    tail=700))
print('--- reinstall wheel:')
print(run(f'{venv}/pip install --quiet --force-reinstall --no-deps '
          '/tmp/cantera-oracle/src/build/python/dist/Cantera-*.whl'))
print('--- import check:')
print(run(f'{venv}/python -c '
          '"import cantera as ct; print(\'OK\', ct.__version__, ct.__git_commit__)"',
          tail=600))
