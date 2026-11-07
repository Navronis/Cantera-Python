path = 'outputs/cantera-python/cantera_python/surface.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

old_str = "net_prod = np.asarray(self.net_production_rates(), dtype=float)"
new_str = "npr = self.net_production_rates; net_prod = np.asarray(npr() if callable(npr) else npr, dtype=float)"

if old_str in content:
    content = content.replace(old_str, new_str, 1)
    print("Fixed net_prod call")
else:
    print("old_str not found")

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
