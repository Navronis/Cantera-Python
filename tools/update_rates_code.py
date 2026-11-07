import sys

def main():
    with open('cantera_python/rates.py', 'r', encoding='utf-8') as f:
        text = f.read()

    old_block = '''rate_classes = [
    ArrheniusRate, BlowersMaselRate, ChebyshevRate,
    ConstantVibrationalRelaxationRate, CustomFunc1Rate,
    ElectronCollisionPlasmaRate, LindemannRate, LinearBurkeRate,
    PlogRate, SriRate, TroeRate, TsangRate, TwoTempPlasmaRate,
    VibrationalRelaxationRate, CastelaVibrationalRelaxationRate,
    MultiStateResolvedVibrationalRelaxationRate,
    StarikovskiyVibrationalRelaxationRate
]'''

    new_block = '''rate_classes = [
    ArrheniusRate, BlowersMaselRate, ChebyshevRate,
    ConstantVibrationalRelaxationRate, CustomFunc1Rate,
    ElectronCollisionPlasmaRate, LindemannRate, LinearBurkeRate,
    PlogRate, SriRate, TroeRate, TsangRate, TwoTempPlasmaRate,
    VibrationalRelaxationRate, CastelaVibrationalRelaxationRate,
    MultiStateResolvedVibrationalRelaxationRate,
    StarikovskiyVibrationalRelaxationRate,
    InterfaceArrheniusRate, InterfaceBlowersMaselRate,
    StickingArrheniusRate, StickingBlowersMaselRate
]'''

    if old_block not in text:
        print("ERROR: old_block not found in rates.py")
        sys.exit(1)

    updated = text.replace(old_block, new_block, 1)
    with open('cantera_python/rates.py', 'w', encoding='utf-8') as f:
        f.write(updated)
    print("SUCCESS: added surface rate classes to rate_classes")

if __name__ == '__main__':
    main()
