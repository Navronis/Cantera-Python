with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = '''    def usesElectrochemistry(self, kin=None) -> bool:
        if kin is None:
            return False
        n_phases = kin.nPhases() if hasattr(kin, "nPhases") else 1
        e_counter = [0.0] * n_phases
        for name, stoich in self._products.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] += stoich * chg
        for name, stoich in self._reactants.items():
            kkin = kin.kineticsSpeciesIndex(name) if hasattr(kin, "kineticsSpeciesIndex") else kin.species_names.index(name)
            iph = kin.speciesPhaseIndex(kkin) if hasattr(kin, "speciesPhaseIndex") else 0
            p = kin.thermo(iph) if hasattr(kin, "thermo") else kin
            kphase = p.speciesIndex(name) if hasattr(p, "speciesIndex") else p.species_names.index(name)
            chg = p.charge(kphase) if hasattr(p, "charge") else 0.0
            e_counter[iph] -= stoich * chg
        return any(abs(e) > 0.001 for e in e_counter)'''

new_block = '''    def usesElectrochemistry(self, kin=None) -> bool:
        if kin is None:
            return False
        n_phases = kin.nPhases() if callable(getattr(kin, "nPhases", None)) else (getattr(kin, "n_phases", 1) if hasattr(kin, "n_phases") else 1)
        e_counter = [0.0] * max(1, n_phases)
        for name, stoich in self._products.items():
            if hasattr(kin, "kineticsSpeciesIndex") and callable(kin.kineticsSpeciesIndex):
                kkin = kin.kineticsSpeciesIndex(name)
            elif hasattr(kin, "kinetics_species_index") and callable(kin.kinetics_species_index):
                kkin = kin.kinetics_species_index(name)
            elif hasattr(kin, "species_names") and name in kin.species_names:
                kkin = kin.species_names.index(name)
            else:
                kkin = 0
            if hasattr(kin, "speciesPhaseIndex") and callable(kin.speciesPhaseIndex):
                iph = kin.speciesPhaseIndex(kkin)
            elif hasattr(kin, "species_phase_index") and callable(kin.species_phase_index):
                iph = kin.species_phase_index(kkin)
            else:
                iph = 0
            if hasattr(kin, "phase") and callable(kin.phase):
                p = kin.phase(iph)
            elif hasattr(kin, "thermo") and callable(kin.thermo):
                p = kin.thermo(iph)
            elif hasattr(kin, "_phases"):
                p = kin._phases[iph] if iph < len(kin._phases) else kin
            else:
                p = getattr(kin, "thermo", kin)
            if hasattr(p, "speciesIndex") and callable(p.speciesIndex):
                kphase = p.speciesIndex(name)
            elif hasattr(p, "species_names") and name in p.species_names:
                kphase = p.species_names.index(name)
            else:
                kphase = 0
            if hasattr(p, "charge") and callable(p.charge):
                chg = p.charge(kphase)
            elif hasattr(p, "charges") and kphase < len(p.charges):
                chg = p.charges[kphase]
            else:
                chg = 0.0
            if iph < len(e_counter):
                e_counter[iph] += stoich * chg
        for name, stoich in self._reactants.items():
            if hasattr(kin, "kineticsSpeciesIndex") and callable(kin.kineticsSpeciesIndex):
                kkin = kin.kineticsSpeciesIndex(name)
            elif hasattr(kin, "kinetics_species_index") and callable(kin.kinetics_species_index):
                kkin = kin.kinetics_species_index(name)
            elif hasattr(kin, "species_names") and name in kin.species_names:
                kkin = kin.species_names.index(name)
            else:
                kkin = 0
            if hasattr(kin, "speciesPhaseIndex") and callable(kin.speciesPhaseIndex):
                iph = kin.speciesPhaseIndex(kkin)
            elif hasattr(kin, "species_phase_index") and callable(kin.species_phase_index):
                iph = kin.species_phase_index(kkin)
            else:
                iph = 0
            if hasattr(kin, "phase") and callable(kin.phase):
                p = kin.phase(iph)
            elif hasattr(kin, "thermo") and callable(kin.thermo):
                p = kin.thermo(iph)
            elif hasattr(kin, "_phases"):
                p = kin._phases[iph] if iph < len(kin._phases) else kin
            else:
                p = getattr(kin, "thermo", kin)
            if hasattr(p, "speciesIndex") and callable(p.speciesIndex):
                kphase = p.speciesIndex(name)
            elif hasattr(p, "species_names") and name in p.species_names:
                kphase = p.species_names.index(name)
            else:
                kphase = 0
            if hasattr(p, "charge") and callable(p.charge):
                chg = p.charge(kphase)
            elif hasattr(p, "charges") and kphase < len(p.charges):
                chg = p.charges[kphase]
            else:
                chg = 0.0
            if iph < len(e_counter):
                e_counter[iph] -= stoich * chg
        return any(abs(e) > 0.001 for e in e_counter)'''

text_norm = text.replace('\r\n', '\n')
if old_block not in text_norm:
    print('ERROR: old_block not found in kinetics.py')
else:
    updated = text_norm.replace(old_block, new_block, 1)
    with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
        f.write(updated)
    print('SUCCESS: updated usesElectrochemistry in kinetics.py')
