with open('cantera_python/rates.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''def _rate_subType(self) -> str:
    subtype_name = type(self).__name__
    return subtype_name'''

addition = '''_NATIVE_SUBTYPE_MAP = {
    "LindemannRate": "Lindemann",
    "TroeRate": "Troe",
    "SriRate": "SRI",
    "TsangRate": "Tsang",
}

def _rate_subType(self) -> str:
    return _NATIVE_SUBTYPE_MAP.get(type(self).__name__, "")'''

assert target in text
updated = text.replace(target, addition, 1)
with open('cantera_python/rates.py', 'w', encoding='utf-8') as f:
    f.write(updated)
print('SUCCESS: updated _rate_subType in rates.py')
