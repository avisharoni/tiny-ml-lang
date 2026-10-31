# tiny-ml-lang

![tests](https://github.com/avisharoni/tiny-ml-lang/actions/workflows/tests.yml/badge.svg)

**Area:** Programming Languages

A tiny ML-style functional language, about 250 lines: tokenizer, recursive-descent parser, **Hindley-Milner type inference** (Algorithm W with let-polymorphism and occurs check), and an environment-based evaluator with closures and tail-call looping.

```python
from tinyml import run, infer
run("let rec f n = if n < 2 then 1 else n * f (n - 1) in f 10")  # (3628800, 'int')
infer("fun f -> fun g -> fun x -> f (g x)")  # "('a -> 'b) -> ('c -> 'a) -> 'c -> 'b"
```

Tests cover precedence, let-polymorphism, recursion, principal types for compose/twice, rejection of ill-typed programs (including `fun x -> x x`), and a 50,000-deep tail-recursive loop.

## Run

```bash
pip install -r requirements.txt
python -m pytest -q
```

## Limitations

Only ints, bools, and functions; no data types or pattern matching.

MIT licensed.
