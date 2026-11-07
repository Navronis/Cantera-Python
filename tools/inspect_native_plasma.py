import os
os.environ["CANTERA_DATA"] = "/home/nebula/cantera-pinned-oracle/data:/home/nebula/cantera-pinned-oracle/test/data"
import cantera as ct
sol = ct.Solution("oxygen-plasma.yaml", transport_model=None)
print("mean_energy:", sol.mean_electron_energy)
print("levels:", list(sol.electron_energy_levels))
print("dist:", list(sol.electron_energy_distribution))
print("dist_type:", sol.electron_energy_distribution_type)
