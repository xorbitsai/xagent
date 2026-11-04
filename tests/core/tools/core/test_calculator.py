from xagent.core.tools.core.calculator import calculator


def test_calculator_basic_arithmetic():
    """Test basic arithmetic operations."""
    assert calculator("2 + 2") == "The answer is: 4"
    assert calculator("5 - 3") == "The answer is: 2"
    assert calculator("4 * 6") == "The answer is: 24"
    assert calculator("10 / 2") == "The answer is: 5.0"


def test_calculator_complex_expressions():
    """Test complex mathematical expressions."""
    assert calculator("sqrt(25) + 5") == "The answer is: 10.0"
    assert calculator("2 ** 3") == "The answer is: 8"
    assert calculator("10 % 3") == "The answer is: 1"
    assert calculator("sin(0)") == "The answer is: 0.0"


def test_calculator_invalid_expression():
    """Test invalid mathematical expressions."""
    result = calculator("invalid expression")
    assert result.startswith("Error: Invalid expression")
    assert "Could not evaluate" in result


def test_calculator_division_by_zero():
    """Test division by zero handling."""
    result = calculator("5 / 0")
    assert result.startswith("Error: Invalid expression")


def test_calculator_whitespace_handling():
    """Test whitespace handling in expressions."""
    assert calculator("  2 + 2  ") == "The answer is: 4"
    assert calculator("2+2") == "The answer is: 4"


def test_calculator_parentheses():
    """Test expressions with parentheses."""
    assert calculator("(2 + 3) * 4") == "The answer is: 20"
    assert calculator("2 * (3 + 4)") == "The answer is: 14"


def test_calculator_decimal_numbers():
    """Test decimal number operations."""
    assert calculator("3.14 * 2") == "The answer is: 6.28"
    assert calculator("10.5 / 2") == "The answer is: 5.25"


def test_calculator_negative_numbers():
    """Test negative number operations."""
    assert calculator("-5 + 3") == "The answer is: -2"
    assert calculator("10 * -2") == "The answer is: -20"


def test_calculator_multiple_operations():
    """Test expressions with multiple operations."""
    assert calculator("2 + 3 * 4") == "The answer is: 14"
    assert calculator("(2 + 3) * 4 - 1") == "The answer is: 19"
