import numexpr


def calculator(expression: str) -> str:
    """
    A powerful calculator. Use this tool to evaluate any mathematical expression.
    It supports arithmetic operations (+, -, *, /), exponentiation (**), modulo (%),
    and more complex functions like sqrt(), sin(), cos(), etc.

    Example:
    User: What is the square root of 25 plus 5?
    LLM calls calculator with: `sqrt(25) + 5`
    """
    try:
        # Use the safe numexpr.evaluate() function
        result = numexpr.evaluate(expression.strip())

        # Return the result as a string
        return f"The answer is: {result}"

    except Exception as e:
        # If an error occurs, return a descriptive message
        return f"Error: Invalid expression '{expression}'. Could not evaluate. Please provide a valid mathematical expression. Details: {e}"
