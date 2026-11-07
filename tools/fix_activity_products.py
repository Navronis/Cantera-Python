with open('cantera_python/kinetics.py', 'r', encoding='utf-8') as f:
    c = f.read()

bad_func = '''    def _activity_products(self, conc, orders, general=None):
        conc = np.asarray(conc, dtype=float)
        orders = np.asarray(orders, dtype=float)
        if general is not None:
            general = np.asarray(general, dtype=bool)
            rates = np.zeros(orders.shape[0], dtype=float)
            for i in range(orders.shape[0]):
                o = orders[i]
                if not general[i]:
                    p = 1.0
                    for c_val, order_val in zip(conc, o):
                        if order_val == 1.0:
                            p *= c_val
                        elif order_val == 2.0:
                            p *= (c_val * c_val)
                        elif order_val == 3.0:
                            p *= (c_val * c_val * c_val)
                    rates[i] = p
                else:
                    active_idx = np.where(o != 0.0)[0]
                    if np.any(conc[active_idx] <= 0.0):
                        rates[i] = 0.0
                    else:
                        rates[i] = np.prod(conc[active_idx] ** o[active_idx])
            return rates
    def _legacy_activity_products(self, conc, orders):'''

good_func = '''    def _activity_products(self, conc, orders, general=None):
        conc = np.asarray(conc, dtype=float)
        orders = np.asarray(orders, dtype=float)
        if general is None:
            general = getattr(self, "_general_reactants", None)
        if general is None:
            general = self._general_stoich_products(orders, orders)
        general = np.asarray(general, dtype=bool)
        rates = np.zeros(orders.shape[0], dtype=float)
        for i in range(orders.shape[0]):
            o = orders[i]
            if not general[i]:
                p = 1.0
                for c_val, order_val in zip(conc, o):
                    if order_val == 1.0:
                        p *= c_val
                    elif order_val == 2.0:
                        p *= (c_val * c_val)
                    elif order_val == 3.0:
                        p *= (c_val * c_val * c_val)
                rates[i] = p
            else:
                active_idx = np.where(o != 0.0)[0]
                if np.any(conc[active_idx] <= 0.0):
                    rates[i] = 0.0
                else:
                    rates[i] = np.prod(conc[active_idx] ** o[active_idx])
        return rates'''

assert bad_func in c, "bad_func not found in kinetics.py"
c = c.replace(bad_func, good_func)
with open('cantera_python/kinetics.py', 'w', encoding='utf-8') as f:
    f.write(c)
print("Successfully fixed _activity_products.")
