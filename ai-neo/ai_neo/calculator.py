"""A safe, exact arithmetic evaluator built on `ast`. It never calls eval() or exec().

Integers and decimal literals are evaluated exactly with `fractions.Fraction`. Functions such as
`sin` or `log` return floats. Only the operators and names listed here are accepted.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from decimal import Decimal, localcontext
from fractions import Fraction

Number = Fraction | float

MAX_EXPR_LEN = 500
MAX_DEPTH = 40
MAX_EXPONENT = 10_000
MAX_BITS = 100_000
SCIENTIFIC_ABOVE_BITS = 13_000


class CalcError(ValueError):
    """The expression is invalid, unsupported, or too expensive to evaluate."""


def _num(value: int | float) -> Number:
    if isinstance(value, int):
        return Fraction(value)
    try:
        return Fraction(repr(value))
    except (ValueError, OverflowError):
        raise CalcError("number out of range") from None


def _bits(value: Number) -> int:
    if isinstance(value, float):
        return 0
    return max(value.numerator.bit_length(), value.denominator.bit_length())


def _checked(value: Number) -> Number:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CalcError("result is not a finite number")
    elif _bits(value) > MAX_BITS:
        raise CalcError("result is too large")
    return value


def _as_float(value: Number) -> float:
    try:
        return float(value)
    except OverflowError:
        raise CalcError("number too large for floating-point math") from None


def _is_int(value: Number) -> bool:
    return isinstance(value, Fraction) and value.denominator == 1


def _int_arg(value: Number, name: str, lo: int = 0, hi: int = 10_000) -> int:
    if not _is_int(value):
        raise CalcError(f"{name} needs an integer")
    number = int(value)  # type: ignore[arg-type]
    if not lo <= number <= hi:
        raise CalcError(f"{name} argument must be between {lo} and {hi}")
    return number


def _pow(a: Number, b: Number) -> Number:
    if isinstance(a, Fraction) and _is_int(b):
        exponent = int(b)  # type: ignore[arg-type]
        if abs(exponent) > MAX_EXPONENT:
            raise CalcError(f"exponent too large (limit {MAX_EXPONENT})")
        if a == 0 and exponent < 0:
            raise CalcError("division by zero")
        if _bits(a) * abs(exponent) > MAX_BITS:
            raise CalcError("result is too large")
        return a**exponent
    x, y = _as_float(a), _as_float(b)
    if x < 0 and y != math.floor(y):
        raise CalcError("negative base with a fractional exponent is not real")
    try:
        result = x**y
    except (OverflowError, ZeroDivisionError):
        raise CalcError("power overflow or division by zero") from None
    if isinstance(result, complex):
        raise CalcError("result is not a real number")
    return result


def _binary(op: type[ast.operator], a: Number, b: Number) -> Number:
    if op is ast.Pow:
        return _checked(_pow(a, b))
    if op in (ast.Div, ast.FloorDiv, ast.Mod) and b == 0:
        raise CalcError("division by zero")
    table: dict[type[ast.operator], Callable[[object, object], object]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
    }
    if op not in table:
        raise CalcError(f"operator {op.__name__} is not supported")
    if isinstance(a, float) or isinstance(b, float):
        result = table[op](_as_float(a), _as_float(b))
    else:
        result = table[op](a, b)
    if isinstance(result, int):
        result = Fraction(result)
    return _checked(result)  # type: ignore[arg-type]


def _sqrt(x: Number) -> Number:
    if isinstance(x, Fraction):
        if x < 0:
            raise CalcError("sqrt of a negative number")
        n, d = math.isqrt(x.numerator), math.isqrt(x.denominator)
        if n * n == x.numerator and d * d == x.denominator:
            return Fraction(n, d)
    return math.sqrt(_as_float(x))


def _float_fn(fn: Callable[..., float]) -> Callable[..., Number]:
    def wrapper(*args: Number) -> Number:
        try:
            return fn(*[_as_float(a) for a in args])
        except (ValueError, OverflowError) as exc:
            raise CalcError(f"math domain error: {exc}") from None

    return wrapper


def _round(x: Number, digits: Number = Fraction(0)) -> Number:
    places = _int_arg(digits, "round", -30, 30)
    if isinstance(x, float):
        return round(x, places)
    result = round(x, places)
    return Fraction(result)


def _floor(x: Number) -> Number:
    return Fraction(math.floor(x))


def _ceil(x: Number) -> Number:
    return Fraction(math.ceil(x))


def _minmax(pick: Callable[..., Number]) -> Callable[..., Number]:
    def wrapper(*args: Number) -> Number:
        return pick(args)

    return wrapper


def _log(x: Number, base: Number | None = None) -> Number:
    try:
        if base is None:
            return math.log(_as_float(x))
        return math.log(_as_float(x), _as_float(base))
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise CalcError(f"math domain error: {exc}") from None


def _int_fn(fn: Callable[..., int], hi: int) -> Callable[..., Number]:
    def wrapper(*args: Number) -> Number:
        ints = [_int_arg(a, fn.__name__, 0, hi) for a in args]
        return _checked(Fraction(fn(*ints)))

    return wrapper


# name -> (implementation, min args, max args)
FUNCTIONS: dict[str, tuple[Callable[..., Number], int, int]] = {
    "abs": (lambda x: abs(x), 1, 1),
    "round": (_round, 1, 2),
    "min": (_minmax(min), 1, 20),
    "max": (_minmax(max), 1, 20),
    "floor": (_floor, 1, 1),
    "ceil": (_ceil, 1, 1),
    "sqrt": (_sqrt, 1, 1),
    "exp": (_float_fn(math.exp), 1, 1),
    "log": (_log, 1, 2),
    "log2": (_float_fn(math.log2), 1, 1),
    "log10": (_float_fn(math.log10), 1, 1),
    "sin": (_float_fn(math.sin), 1, 1),
    "cos": (_float_fn(math.cos), 1, 1),
    "tan": (_float_fn(math.tan), 1, 1),
    "asin": (_float_fn(math.asin), 1, 1),
    "acos": (_float_fn(math.acos), 1, 1),
    "atan": (_float_fn(math.atan), 1, 1),
    "atan2": (_float_fn(math.atan2), 2, 2),
    "radians": (_float_fn(math.radians), 1, 1),
    "degrees": (_float_fn(math.degrees), 1, 1),
    "factorial": (_int_fn(math.factorial, 3000), 1, 1),
    "gcd": (_int_fn(math.gcd, 10**30), 2, 2),
    "lcm": (_int_fn(math.lcm, 10**30), 2, 2),
    "comb": (_int_fn(math.comb, 10_000), 2, 2),
    "perm": (_int_fn(math.perm, 10_000), 2, 2),
}
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval(node: ast.AST, depth: int) -> Number:
    if depth > MAX_DEPTH:
        raise CalcError("expression is nested too deeply")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise CalcError(f"only numbers are allowed, not {node.value!r}")
        return _num(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub | ast.UAdd):
        operand = _eval(node.operand, depth + 1)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, depth + 1)
        right = _eval(node.right, depth + 1)
        return _binary(type(node.op), left, right)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        raise CalcError(f"unknown name '{node.id}' (constants: {', '.join(CONSTANTS)})")
    if isinstance(node, ast.Call):
        return _call(node, depth)
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


def _call(node: ast.Call, depth: int) -> Number:
    if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
        raise CalcError(f"unknown function (allowed: {', '.join(FUNCTIONS)})")
    if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
        raise CalcError("keyword and * arguments are not supported")
    fn, low, high = FUNCTIONS[node.func.id]
    if not low <= len(node.args) <= high:
        raise CalcError(f"{node.func.id}() takes {low}-{high} arguments, got {len(node.args)}")
    args = [_eval(a, depth + 1) for a in node.args]
    return _checked(fn(*args))


def safe_eval(expression: str) -> Number:
    """Evaluate an arithmetic expression. Raises CalcError for anything else."""
    if not isinstance(expression, str):
        raise CalcError("expression must be a string")
    text = expression.strip()
    if not text:
        raise CalcError("empty expression")
    if len(text) > MAX_EXPR_LEN:
        raise CalcError(f"expression too long (limit {MAX_EXPR_LEN} characters)")
    if "^" in text:
        raise CalcError("'^' means XOR here; use ** for powers")
    try:
        tree = ast.parse(text, mode="eval")
        return _checked(_eval(tree.body, 0))
    except CalcError:
        raise
    except SyntaxError as exc:
        raise CalcError(f"cannot parse expression: {exc.msg}") from None
    except (RecursionError, MemoryError):
        raise CalcError("expression is too complex") from None
    except (ValueError, OverflowError, ZeroDivisionError, TypeError) as exc:
        raise CalcError(f"cannot evaluate: {exc}") from None


def _terminating_decimal(value: Fraction) -> str | None:
    """Exact decimal text for fractions whose denominator is 2^a * 5^b with small a and b."""
    den = value.denominator
    twos = fives = 0
    while den % 2 == 0:
        den //= 2
        twos += 1
    while den % 5 == 0:
        den //= 5
        fives += 1
    if den != 1 or max(twos, fives) > 30:
        return None
    with localcontext() as ctx:
        ctx.prec = 200
        return format(Decimal(value.numerator) / Decimal(value.denominator), "f")


def _format_big_int(n: int) -> str:
    sign = "-" if n < 0 else ""
    n = abs(n)
    shift = n.bit_length() - 100
    log10 = math.log10(n >> shift) + shift * math.log10(2)
    exponent = int(log10)
    mantissa = 10 ** (log10 - exponent)
    return f"{sign}~{mantissa:.12f}e+{exponent} (about {exponent + 1} digits)"


def format_number(value: Number) -> str:
    if isinstance(value, float):
        return repr(value)
    if value.denominator == 1:
        n = value.numerator
        if n.bit_length() > SCIENTIFIC_ABOVE_BITS:
            return _format_big_int(n)
        return str(n)
    exact = None if _bits(value) > SCIENTIFIC_ABOVE_BITS else _terminating_decimal(value)
    if exact is not None:
        return exact
    with localcontext() as ctx:
        ctx.prec = 15
        approx = Decimal(value.numerator) / Decimal(value.denominator)
    if _bits(value) > SCIENTIFIC_ABOVE_BITS:
        return f"about {approx} (a very large fraction)"
    return f"{value.numerator}/{value.denominator} (about {approx})"
