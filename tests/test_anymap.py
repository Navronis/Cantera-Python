import pytest
import cantera_python as ct
from cantera_python.anymap import AnyBase, AnyValue, AnyMap, Application, AnyMapError

def test_anyvalue_scalars_and_conversion():
    v_num = AnyValue(42.5)
    assert v_num.asDouble() == 42.5
    assert v_num.type_name() == "double"

    v_int = AnyValue(10)
    assert v_int.asInt() == 10
    assert v_int.type_name() == "int"

    v_bool = AnyValue(True)
    assert v_bool.asBool() is True
    assert v_bool.type_name() == "bool"

    v_str = AnyValue("cantera")
    assert v_str.asString() == "cantera"
    assert v_str.type_name() == "string"

    v_vec = AnyValue([1.0, 2.0, 3.0])
    assert v_vec.asVector() == [1.0, 2.0, 3.0]
    assert v_vec.type_name() == "vector"

def test_anymap_nesting_and_serialization():
    m = AnyMap()
    m["name"] = "test-phase"
    m["temperature"] = 300.0
    m["species"] = ["H2", "O2", "H2O"]

    assert m.hasKey("name")
    assert m.getString("name") == "test-phase"
    assert m.getDouble("temperature") == 300.0
    assert m["species"].asVector() == ["H2", "O2", "H2O"]

    # Nested map
    m["nested"]["nested_key"] = 123
    assert m["nested"]["nested_key"].asInt() == 123

    # YAML serialization
    yaml_str = m.to_yaml()
    assert "test-phase" in yaml_str
    m2 = AnyMap.from_yaml(yaml_str)
    assert m2.getString("name") == "test-phase"
    assert m2.getDouble("temperature") == 300.0

def test_application_singleton():
    app = Application.Instance()
    assert app is not None
    dirs = app.getDataDirectories()
    assert len(dirs) >= 1

    app.printlog("Testing log")
    log_content = app.getLog()
    assert "Testing log" in log_content

    # Find standard file
    gri = app.findInputFile("gri30.yaml")
    assert gri is not None
