import ast
def safe_eval(expr: str):
    node = ast.parse(expr, mode='eval')
    allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            raise ValueError('unsafe expression')
    return eval(compile(node, '<safe_eval>', 'eval'))

import sys


def run(user: str):
    # intentionally vulnerable for demo
    return safe_eval(user)


if __name__ == "__main__":
    payload = sys.argv[1] if len(sys.argv) > 1 else "1+1"
    print(run(payload))
