from __future__ import annotations

import pytest

from ai_neo.calculator import CalcError, format_number, safe_eval


@pytest.mark.parametrize(
    ("expr", "shown"),
    [
        ("2+2*3", "8"),
        ("12345*6789", "83810205"),
        ("(1+2)**3 - 4", "23"),
        ("-2**2", "-4"),
        ("17 % 5 + 17 // 5", "5"),
        ("0.1 + 0.2", "0.3"),
        ("1/3", "1/3 (about 0.333333333333333)"),
        ("7/4", "1.75"),
        ("sqrt(16)", "4"),
        ("sqrt(9/16)", "0.75"),
        ("factorial(10)", "3628800"),
        ("max(3, 9, 4) - min(3, 9, 4)", "6"),
        ("round(7/2)", "4"),
        ("gcd(84, 36)", "12"),
        ("2**-2", "0.25"),
        ("abs(-3.5)", "3.5"),
    ],
)
def test_exact_results(expr: str, shown: str) -> None:
    assert format_number(safe_eval(expr)) == shown


def test_float_functions() -> None:
    assert float(safe_eval("sqrt(2)")) == pytest.approx(1.4142135623730951)
    assert float(safe_eval("sin(pi/2)")) == pytest.approx(1.0)
    assert float(safe_eval("log(e**2)")) == pytest.approx(2.0)
    assert float(safe_eval("2**0.5")) == pytest.approx(1.41421356)


def test_huge_but_allowed_results_format_safely() -> None:
    assert len(format_number(safe_eval("2**1000"))) == 302
    assert "digits" in format_number(safe_eval("2**9000 * 2**9000"))
    assert "very large" in format_number(safe_eval("(2**9000 * 2**9000 + 1) / 3**9000"))


HOSTILE = [
    "__import__('os').system('echo pwned')",
    "().__class__.__bases__[0].__subclasses__()",
    "open('/etc/passwd').read()",
    "[x for x in range(10**9)]",
    "(lambda: 1)()",
    "exec('1')",
    "eval('1')",
    "sqrt.__class__",
    "(1).__add__(2)",
    "1 if True else 2",
    "(x := 5)",
    "f'{1}'",
    "'a' * 10**9",
    "b'abc'",
    "1 < 2",
    "True",
    "None",
    "[1, 2, 3]",
    "{1: 2}",
    "x",
    "import os",
    "a.b",
    "1 ^ 2",
    "1 << 100",
    "~5",
    "not 1",
    "9**9**9",
    "10**10000000",
    "2**100000",
    "10**9999 * 10**9999 * 10**9999 * 10**9999",
    "factorial(10**6)",
    "factorial(-1)",
    "comb(10**9, 5)",
    "1/0",
    "1 // 0",
    "5 % 0",
    "0 ** -1",
    "0.0 ** -1",
    "(-8) ** 0.5",
    "1e999",
    "2.0 ** 100000",
    "exp(1e6)",
    "log(0)",
    "sqrt(-1)",
    "sqrt()",
    "sqrt(1, 2)",
    "sqrt(x=4)",
    "max(*[1, 2])",
    "print(1)",
    "",
    "   ",
    "(" * 500 + "1" + ")" * 500,
    "1+" * 400 + "1",
    "\x00",
    "1;2",
    "1\n2",
]


@pytest.mark.parametrize("expr", HOSTILE)
def test_hostile_expressions_raise_calc_error(expr: str) -> None:
    with pytest.raises(CalcError):
        safe_eval(expr)


def test_deeply_nested_is_rejected_quickly() -> None:
    with pytest.raises(CalcError):
        safe_eval("-" * 400 + "1")


def test_non_string_input() -> None:
    with pytest.raises(CalcError):
        safe_eval(None)  # type: ignore[arg-type]
