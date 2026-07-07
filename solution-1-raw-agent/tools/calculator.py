"""calculator tool - pure local code (standard library).

Safely evaluates arithmetic by parsing the expression into an AST (no eval).
The tool's logic and its OpenAI schema live together in one module.
"""
import ast
import operator

# Whitelist of allowed operations: AST node type -> function from the operator module.
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


def _eval_node(node):
    """Recursively evaluate a tree node, allowing arithmetic only."""
    if isinstance(node, ast.Constant):          # number (leaf of the tree)
        return node.value
    if isinstance(node, ast.BinOp):             # a <op> b
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):           # -a
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("invalid expression")      # anything else -> refuse (safety)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '23 * 47'."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as exc:
        return f"calculation error: {exc}"


# Tool description for the model (OpenAI tools schema).
CALCULATOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluates an arithmetic expression. Pass ONLY pure math in "
                       "Python syntax: numbers and operators + - * / ** ( ). No words "
                       "and no '%' sign; convert percentages to multiplication, e.g. "
                       "'15% of 2400' should be passed as '2400 * 0.15'. Use for any "
                       "precise calculation instead of computing in your head.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "An arithmetic expression, e.g. '23 * 47'",
                }
            },
            "required": ["expression"],
        },
    },
}
