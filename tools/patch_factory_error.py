with open("cantera_python/transport.py", "r", encoding="utf-8") as f:
    text = f.read()

target = """        if model == "none":
            return Transport(phase)
        cls_tr = self._models.get(model, MixTransport)"""

replace = """        if model == "none":
            return Transport(phase)
        if model not in self._models:
            raise CanteraError("TransportFactory::newTransport", f"Unknown transport model: {model}")
        cls_tr = self._models[model]"""

if "\r\n" in text:
    target = target.replace("\n", "\r\n")
    replace = replace.replace("\n", "\r\n")

if target in text:
    text = text.replace(target, replace, 1)
    with open("cantera_python/transport.py", "w", encoding="utf-8") as f:
        f.write(text)
    print("TransportFactory updated with CanteraError check")
else:
    print("Warning: target not found")
