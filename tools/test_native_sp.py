import cantera as ct
from pathlib import Path

TEST_DATA = Path('/mnt/c/Users/Dell/Documents/Codex/2026-09-11/https-github-com-cantera-cantera-https/outputs/cantera-original/test/data')
pr = ct.Solution(str(TEST_DATA / 'thermo-models.yaml'), 'CO2-PR')
pr.TP = 367.19487050431525, 5000000.0
print('Before step native: density =', pr.density)
pr.T = 272.53
pr.P = 4894025.902187561
print('After step in native: density =', pr.density)
