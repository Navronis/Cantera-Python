# Update test_canonical_inventory.py
with open("tests/test_canonical_inventory.py", "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace(
    '                "NOT_PORTED": 2833,\n                "PINNED_NATIVE_VALIDATED": 1737,\n                "NON_RUNTIME_CPP_ONLY_PROVEN": 759,\n                "PARTIAL": 455,',
    '                "NOT_PORTED": 2779,\n                "PINNED_NATIVE_VALIDATED": 1852,\n                "NON_RUNTIME_CPP_ONLY_PROVEN": 759,\n                "PARTIAL": 394,'
)
c = c.replace(
    'sum(row["current_status"] in final_statuses for row in self.rows), 2496)',
    'sum(row["current_status"] in final_statuses for row in self.rows), 2611)'
)
with open("tests/test_canonical_inventory.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Updated test_canonical_inventory.py")

# Update test_symbol_closure_dashboard.py
with open("tests/test_symbol_closure_dashboard.py", "r", encoding="utf-8") as f:
    c = f.read()

c = c.replace("self.assertEqual(dashboard['fully_dispositioned'], 2496)", "self.assertEqual(dashboard['fully_dispositioned'], 2611)")
c = c.replace("self.assertEqual(dashboard['remaining_without_final_disposition'], 3288)", "self.assertEqual(dashboard['remaining_without_final_disposition'], 3173)")
with open("tests/test_symbol_closure_dashboard.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Updated test_symbol_closure_dashboard.py")
