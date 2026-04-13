from xagent.core.utils.type_check import ensure_list, is_list_of_type


def test_ensure_list():
    # 1. Test None
    assert ensure_list(None) is None

    # 2. Test plain list of strings
    assert ensure_list(["a", "b"]) == ["a", "b"]

    # 3. Test plain list of mixed types (should be cast to string)
    assert ensure_list([1, 2.5, True]) == ["1", "2.5", "True"]

    # 4. Test stringified JSON array
    assert ensure_list('["knowledge", "file"]') == ["knowledge", "file"]

    # 5. Test stringified JSON array with non-string elements
    assert ensure_list('[1, true, "text"]') == ["1", "True", "text"]

    # 6. Test valid JSON but not an array (e.g. object)
    assert ensure_list('{"key": "value"}') == ['{"key": "value"}']

    # 7. Test invalid JSON string (should be wrapped in a list)
    assert ensure_list("not a json array") == ["not a json array"]

    # 8. Test plain string that looks like a keyword
    assert ensure_list("knowledge") == ["knowledge"]


def test_empty_list():
    assert not is_list_of_type(str, [])
    assert not is_list_of_type(int, [])


def test_list_of_strings():
    assert is_list_of_type(str, ["a", "b"])
    assert not is_list_of_type(int, ["a", "b"])


def test_list_of_ints():
    assert is_list_of_type(int, [1, 2, 3])
    assert not is_list_of_type(str, [1, 2, 3])


def test_list_of_floats():
    assert is_list_of_type(float, [1.0, 2.0, 3.0])
    assert not is_list_of_type(int, [1.0, 2.0, 3.0])


def test_mixed_list():
    assert not is_list_of_type(int, [1, "a"])
    assert not is_list_of_type(int, ["a", "1"])


def test_single_element():
    assert is_list_of_type(int, [42])
    assert is_list_of_type(str, ["hello"])


def test_custom_class():
    class MyClass:
        pass

    obj = MyClass()
    assert is_list_of_type(MyClass, [obj])
    assert not is_list_of_type(str, [obj])
