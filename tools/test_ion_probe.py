import cantera as ct

sol = ct.Solution("/home/nebula/cantera-pinned-oracle/test/data/ch4_ion.yaml", "gas", "ionized-gas")
sol.TP = 300.0, 101325.0
sol.X = "CH4:0.095, O2:0.19, N2:0.715"
d = sol.transport.binary_diffusion_coefficients
k_o2m = sol.species_index("O2-")
for j_name in ["O2", "N2", "CH4"]:
    j = sol.species_index(j_name)
    print(f"native diff(O2-, {j_name}): {d[k_o2m, j]}")

mobi = sol.transport.mobilities
print(f"native mobi(O2-): {mobi[k_o2m]}")
mix_diff = sol.transport.mix_diff_coeffs
print(f"native mix_diff(O2-): {mix_diff[k_o2m]}")
