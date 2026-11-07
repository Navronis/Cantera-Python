p1 = 'outputs/cantera-python/tests/test_symbol_closure_dashboard.py'
with open(p1, 'r', encoding='utf-8') as f:
    t1 = f.read()
t1 = t1.replace("dashboard['fully_dispositioned'], 2611", "dashboard['fully_dispositioned'], 2771")
t1 = t1.replace("dashboard['remaining_without_final_disposition'], 3173", "dashboard['remaining_without_final_disposition'], 3013")
with open(p1, 'w', encoding='utf-8') as f:
    f.write(t1)

p2 = 'outputs/cantera-python/tests/test_canonical_inventory.py'
with open(p2, 'r', encoding='utf-8') as f:
    t2 = f.read()
t2 = t2.replace('"NOT_PORTED": 2779', '"NOT_PORTED": 2630')
t2 = t2.replace('"PINNED_NATIVE_VALIDATED": 1852', '"PINNED_NATIVE_VALIDATED": 2012')
t2 = t2.replace('"PARTIAL": 394', '"PARTIAL": 383')
t2 = t2.replace('sum(row["current_status"] in final_statuses for row in self.rows), 2611)', 'sum(row["current_status"] in final_statuses for row in self.rows), 2771)')
with open(p2, 'w', encoding='utf-8') as f:
    f.write(t2)

print('Updated both test files successfully!')
