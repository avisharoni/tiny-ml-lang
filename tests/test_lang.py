import pytest
from tinyml import run, infer
from tinyml.core import TypeError_

def test_arith_and_precedence():
    assert run("1 + 2 * 3")[0] == 7 and run("(1 + 2) * 3") == (9, "int")

def test_let_poly_identity():
    v, t = run("let id x = x in if id true then id 1 else id 0")
    assert v == 1 and t == "int"

def test_recursion_factorial_and_fib():
    assert run("let rec f n = if n < 2 then 1 else n * f (n - 1) in f 10")[0] == 3628800
    assert run("let rec fib n = if n < 2 then n else fib (n - 1) + fib (n - 2) in fib 15")[0] == 610

def test_inferred_types():
    assert infer("fun x -> x") == "'a -> 'a"
    assert infer("fun f -> fun x -> f (f x)") == "('a -> 'a) -> 'a -> 'a"
    assert infer("fun f -> fun g -> fun x -> f (g x)") == "('a -> 'b) -> ('c -> 'a) -> 'c -> 'b"

def test_type_errors():
    for bad in ["1 + true", "if 1 then 2 else 3", "fun x -> x x", "let f = fun x -> x in f 1 2"]:
        with pytest.raises(TypeError_): infer(bad)

def test_tail_recursion_deep():
    assert run("let rec loop n = fun acc -> if n == 0 then acc else loop (n - 1) (acc + 1) in loop 50000 0")[0] == 50000

def test_zero_literal():
    assert run("0 + 0")[0] == 0
