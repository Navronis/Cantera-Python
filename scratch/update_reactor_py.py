import re
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
reactor_py = repo_root / "cantera_python" / "reactor.py"

with open(reactor_py, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update imports
import_target = "from .numerics import DenseMatrix, _python_l2_norm"
import_replacement = "from .numerics import DenseMatrix, FuncEval, CVodesIntegrator, _python_l2_norm"
if import_target in content:
    content = content.replace(import_target, import_replacement, 1)

# 2. Add componentName, upperBound, lowerBound to ReactorBase
rb_target = """    def clearAbsoluteTolerances(self) -> None:
        self._userAtol = []
        if hasattr(self, '_network') and self._network:
            self._network.setNeedsReinit(True)"""

rb_replacement = """    def clearAbsoluteTolerances(self) -> None:
        self._userAtol = []
        if hasattr(self, '_network') and self._network:
            self._network.setNeedsReinit(True)

    def componentName(self, k: int) -> str:
        if hasattr(self, 'component_name'):
            return self.component_name(k)
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            return str(names[k])
        return f"component_{k}"

    def upperBound(self, k: int) -> float:
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            name = names[k]
            if name in ('mass', 'volume', 'temperature', 'int_energy', 'enthalpy'):
                return 1e300
            return 1.0
        return 1e300

    def lowerBound(self, k: int) -> float:
        names = getattr(self, 'component_names', None)
        if names and k < len(names):
            name = names[k]
            if name in ('mass', 'volume', 'temperature'):
                return 0.0
            elif name in ('int_energy', 'enthalpy'):
                return -1e300
            return -1e-20
        return -1e300"""

content = content.replace(rb_target, rb_replacement, 1)

# 3. Add helper classes _CallableFloat and _CallableInt
helper_code = '''
class _CallableFloat(float):
    """Float subclass that is also callable returning itself."""
    def __call__(self):
        return float(self)


class _CallableInt(int):
    """Int subclass that is also callable returning itself."""
    def __call__(self):
        return int(self)
'''

if "class _CallableFloat" not in content:
    content = content.replace("def _python_dense_solve", helper_code + "\ndef _python_dense_solve", 1)

# Now inspect ReactorNet replacement
with open(reactor_py, "w", encoding="utf-8", newline="\n") as f:
    f.write(content)

print("Updated imports, helper classes, and ReactorBase bounds/componentName.")
