"""Safe parsing of user-typed objective functions such as ``x**2 + 2*x - 3``.

Only a whitelist of names (x, y, numpy math functions, constants) is allowed,
so the expression cannot import modules or touch the file system.
"""

import ast

import numpy as np

ALLOWED_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
    "abs": np.abs, "tanh": np.tanh, "arctan": np.arctan,
}
ALLOWED_CONSTS = {"pi": np.pi, "e": np.e}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
    ast.USub, ast.UAdd,
)


class ExpressionError(ValueError):
    """Raised when a user expression is invalid or unsafe."""


def parse_expression(text):
    """Turn a string into a vectorised objective ``f(P) -> fitness``.

    ``P`` has shape (n_points, dim). Returns ``(func, dim)`` where ``dim`` is
    1 if only ``x`` is used, otherwise 2 (``x`` and ``y``).
    """
    text = text.strip().replace("^", "**")
    if not text:
        raise ExpressionError("Fonksiyon boş olamaz.")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Sözdizimi hatası: {exc.msg}") from exc

    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ExpressionError(f"İzin verilmeyen ifade: {type(node).__name__}")
        if isinstance(node, ast.Call) and not (
            isinstance(node.func, ast.Name) and node.func.id in ALLOWED_FUNCS
        ):
            raise ExpressionError("Yalnızca sin, cos, exp, sqrt, abs... kullanılabilir.")
        if isinstance(node, ast.Name):
            names.add(node.id)

    unknown = names - {"x", "y"} - set(ALLOWED_FUNCS) - set(ALLOWED_CONSTS)
    if unknown:
        raise ExpressionError(f"Bilinmeyen isim: {', '.join(sorted(unknown))}")
    if "x" not in names:
        raise ExpressionError("İfade en az 'x' değişkenini içermeli.")

    dim = 2 if "y" in names else 1
    code = compile(tree, "<fonksiyon>", "eval")

    def func(P):
        P = np.atleast_2d(P)
        env = {"__builtins__": {}, **ALLOWED_FUNCS, **ALLOWED_CONSTS, "x": P[:, 0]}
        if dim == 2:
            env["y"] = P[:, 1]
        with np.errstate(all="ignore"):
            out = eval(code, env)  # noqa: S307 - AST whitelisted above
        out = np.broadcast_to(np.asarray(out, dtype=float), (P.shape[0],)).copy()
        out[~np.isfinite(out)] = 1e12  # penalise invalid points (log(-1) etc.)
        return out

    func(np.zeros((1, dim)))  # smoke test
    return func, dim
