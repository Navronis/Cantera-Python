import sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((root / "tools").resolve()))
from run_native_oracle_runner import run_wave3
if __name__ == "__main__":
    run_wave3()
